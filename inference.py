"""Lightweight inference for the MRI classifier.

The network is split in two:
  * the convolutional "body" runs in LiteRT (TFLite) and returns the last conv feature map;
  * the small classification "head" (pooling + dense layers) runs in NumPy.

Running the head in NumPy lets us compute exact Grad-CAM gradients without shipping
TensorFlow, which keeps the web service well under 512 MB of RAM.
"""
import base64
import io
import json
import os
import threading

import numpy as np
from PIL import Image, ImageOps

RESAMPLING = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def open_image(data):
    """Decode uploaded bytes into an RGB PIL image. Raises ValueError if it is not an image."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # PIL raises many different exception types
        raise ValueError("The uploaded file is not a readable image.") from exc
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def resize_for_model(img, size, resample="nearest"):
    return img.resize((size, size), RESAMPLING[resample])


def to_data_url(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def thumbnail(img, max_side=512):
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.BILINEAR)
    return img


# ---------------------------------------------------------------------------
# Input sanity checks ("is this plausibly a brain MRI slice?")
# ---------------------------------------------------------------------------
def check_input(img):
    """Return (rejections, warnings) for images that are unlikely to be MRI slices.

    These are deliberately simple heuristics: they catch colour photos, blank images,
    drawings and screenshots, but they cannot prove an image is a brain MRI.
    """
    rejections, warnings = [], []
    if min(img.size) < 64:
        rejections.append("The image is too small (minimum 64×64 pixels).")
        return rejections, warnings

    small = np.asarray(thumbnail(img, 256), dtype=np.float32)
    colourfulness = float(np.mean(small.max(axis=2) - small.min(axis=2)))
    gray = small.mean(axis=2)

    if colourfulness > 25:
        rejections.append("The image is in colour. MRI slices are greyscale, so this does not look like an MRI scan.")
    if gray.std() < 12:
        rejections.append("The image is almost uniform (blank). Please upload a real MRI slice.")
    elif len(np.unique(gray.astype(np.uint8) // 4)) < 24:
        rejections.append("The image has too few tones to be a scan (it looks like a drawing or a graphic).")
    if not rejections and np.mean(gray < 30) < 0.05:
        warnings.append("No dark background was detected. Brain MRI slices usually have a black background, "
                        "so check that this is an uncropped axial MRI slice.")
    return rejections, warnings


# ---------------------------------------------------------------------------
# NumPy classification head with exact gradients
# ---------------------------------------------------------------------------
def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class Head:
    """Runs the layers after the conv body. Supports maxpool2, flatten, gap and dense."""

    def __init__(self, ops, weights):
        self.ops = ops
        self.weights = weights

    def forward(self, fmap):
        """fmap: (H, W, C) feature map. Returns logits and the cache needed for backward()."""
        x, cache = fmap, []
        for op in self.ops:
            kind = op["type"]
            if kind == "maxpool2":
                h, w, c = x.shape
                h2, w2 = h // 2, w // 2
                win = x[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, c).transpose(0, 2, 4, 1, 3).reshape(h2, w2, c, 4)
                idx = win.argmax(axis=-1)
                cache.append((x.shape, idx))
                x = win.max(axis=-1)
            elif kind == "flatten":
                cache.append(x.shape)
                x = x.reshape(-1)
            elif kind == "gap":
                cache.append(x.shape)
                x = x.mean(axis=(0, 1))
            elif kind == "dense":
                w, b = self.weights[op["w"]], self.weights[op["b"]]
                pre = x @ w + b
                cache.append(pre)
                x = np.maximum(pre, 0) if op["activation"] == "relu" else pre
            else:
                raise ValueError(f"Unsupported head op: {kind}")
        return x, cache

    def backward(self, grad, cache):
        """Back-propagate d(logit)/d(output) to d(logit)/d(feature map)."""
        for op, saved in zip(reversed(self.ops), reversed(cache)):
            kind = op["type"]
            if kind == "dense":
                if op["activation"] == "relu":
                    grad = grad * (saved > 0)
                grad = self.weights[op["w"]] @ grad
            elif kind == "flatten":
                grad = grad.reshape(saved)
            elif kind == "gap":
                h, w, c = saved
                grad = np.broadcast_to(grad / (h * w), saved).copy()
            elif kind == "maxpool2":
                shape, idx = saved
                h2, w2, c = idx.shape
                full = np.zeros(shape, dtype=grad.dtype)
                dy, dx = np.divmod(idx, 2)
                ys = np.arange(h2)[:, None, None] * 2 + dy
                xs = np.arange(w2)[None, :, None] * 2 + dx
                cs = np.broadcast_to(np.arange(c), idx.shape)
                full[ys, xs, cs] = grad
                grad = full
        return grad


def grad_cam(fmap, grad):
    weights = grad.mean(axis=(0, 1))
    cam = np.maximum((fmap * weights).sum(axis=-1), 0)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def heatmap_overlay(img, cam, alpha=0.55):
    """Blend a jet-coloured Grad-CAM map over the (greyscale) image."""
    base = np.asarray(img, dtype=np.float32) / 255.0
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BILINEAR)
    c = np.asarray(cam_img, dtype=np.float32)[..., None] / 255.0
    jet = np.concatenate([np.clip(1.5 - np.abs(4 * c - k), 0, 1) for k in (3, 2, 1)], axis=-1)
    a = alpha * c
    out = base * (1 - a) + jet * a
    return Image.fromarray(np.uint8(np.clip(out, 0, 1) * 255))


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class Classifier:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "model_meta.json")) as f:
            self.meta = json.load(f)
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=os.path.join(model_dir, self.meta["body"]))
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        weights = np.load(os.path.join(model_dir, self.meta["head_weights"]))
        self.head = Head(self.meta["head_ops"], {k: weights[k] for k in weights.files})
        self.labels = self.meta["labels"]
        self.display_names = self.meta["display_names"]
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.threshold = float(self.meta.get("confidence_threshold", 0.7))
        # A LiteRT interpreter is not thread-safe.
        self._lock = threading.Lock()

    def feature_map(self, img):
        x = np.asarray(resize_for_model(img, self.meta["input_size"], self.meta["resize"]), dtype=np.float32)
        x = x * float(self.meta["input_scale"])
        with self._lock:
            self.interpreter.set_tensor(self.input_index, x[None])
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_index)[0].astype(np.float32)

    def predict(self, img, with_heatmap=True):
        fmap = self.feature_map(img)
        logits, cache = self.head.forward(fmap)
        probs = _softmax(logits / self.temperature)
        top = int(probs.argmax())
        result = {
            "label": self.labels[top],
            "display": self.display_names[top],
            "has_tumor": self.labels[top] != "notumor",
            "confidence": float(probs[top]),
            "confident": bool(probs[top] >= self.threshold),
            "probabilities": [
                {"label": l, "display": d, "probability": float(p)}
                for l, d, p in zip(self.labels, self.display_names, probs)
            ],
        }
        if with_heatmap:
            one_hot = np.zeros_like(logits)
            one_hot[top] = 1.0
            result["cam"] = grad_cam(fmap, self.head.backward(one_hot, cache))
        return result
