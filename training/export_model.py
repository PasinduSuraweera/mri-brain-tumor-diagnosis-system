"""Export a trained Keras classifier into the lightweight format used by the web app.

Produces, inside --out-dir:
  * <name>_body.tflite   conv body (LiteRT), outputs the last conv feature map
  * <name>_head.npz      weights of the classification head (run in NumPy)
  * model_meta.json      labels, preprocessing, head layout, calibration and metrics

The export is verified against the original Keras model (predictions and Grad-CAM
gradients) before anything is written.

Example (the original VGG16 model shipped in this repo):
    python training/export_model.py --model models/model.h5 --name vgg16 \
        --labels glioma meningioma notumor pituitary --input-scale 0.00392156862745098 \
        --resize nearest --metrics-json training/vgg16_metrics.json
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from inference import Head, _softmax  # noqa: E402

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}


def split_model(model, feature_layer=None):
    """Split `model` into (body, head_layers).

    The model must contain a nested backbone (e.g. VGG16 / EfficientNet) followed by
    simple layers. The body ends at the backbone's last conv feature map (or at
    `feature_layer`), and everything after it becomes the head.
    """
    base_pos = next(i for i, l in enumerate(model.layers) if isinstance(l, tf.keras.Model))
    base = model.layers[base_pos]
    if any(not isinstance(l, tf.keras.layers.InputLayer) for l in model.layers[:base_pos]):
        raise ValueError("Layers before the backbone are not exported; put preprocessing inside the "
                         "backbone or express it with --input-scale.")
    if feature_layer:
        feat = base.get_layer(feature_layer)
    else:
        feat = next(l for l in reversed(base.layers)
                    if len(l.output.shape) == 4 and not isinstance(l, tf.keras.layers.MaxPooling2D))
    feat_pos = base.layers.index(feat)
    body = tf.keras.Model(base.input, feat.output, name="body")
    head_layers = base.layers[feat_pos + 1:] + model.layers[base_pos + 1:]
    return body, head_layers


def head_to_ops(head_layers):
    """Translate Keras head layers into NumPy head ops + weight arrays."""
    ops, weights = [], {}
    for i, layer in enumerate(head_layers):
        last = i == len(head_layers) - 1
        if isinstance(layer, tf.keras.layers.Dropout):
            continue
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            if tuple(layer.pool_size) != (2, 2) or tuple(layer.strides) != (2, 2) or layer.padding != "valid":
                raise ValueError(f"Only 2x2/stride-2 valid max pooling is supported ({layer.name})")
            ops.append({"type": "maxpool2"})
        elif isinstance(layer, tf.keras.layers.Flatten):
            ops.append({"type": "flatten"})
        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            ops.append({"type": "gap"})
        elif isinstance(layer, tf.keras.layers.Activation) and last and layer.activation.__name__ == "softmax":
            continue
        elif isinstance(layer, tf.keras.layers.Dense):
            act = layer.activation.__name__
            if act == "softmax" and last:
                act = "linear"  # softmax (with temperature) is applied by the app
            if act not in ("relu", "linear"):
                raise ValueError(f"Unsupported dense activation '{act}' in {layer.name}")
            w, b = layer.get_weights()
            key = f"dense{len(weights) // 2}"
            weights[key + "_w"], weights[key + "_b"] = w.astype(np.float32), b.astype(np.float32)
            ops.append({"type": "dense", "activation": act, "w": key + "_w", "b": key + "_b"})
        else:
            raise ValueError(f"Unsupported head layer {layer.name} ({type(layer).__name__})")
    return ops, weights


def convert_body(body, quantize):
    converter = tf.lite.TFLiteConverter.from_keras_model(body)
    if quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def _ends_with_softmax(head_layers):
    act = getattr(head_layers[-1], "activation", None)
    return act is not None and act.__name__ == "softmax"


def tf_logit_grad(body, head_layers, x, cls):
    """Reference d(logit_cls)/d(feature map) computed with TensorFlow autodiff."""
    fmap = tf.convert_to_tensor(body(x, training=False))
    with tf.GradientTape() as tape:
        tape.watch(fmap)
        h = fmap
        for layer in head_layers[:-1]:
            h = layer(h, training=False)
        last = head_layers[-1]
        if isinstance(last, tf.keras.layers.Dense):
            logits = tf.matmul(h, last.kernel) + last.bias
        else:  # trailing softmax Activation layer
            logits = h
        score = logits[:, cls]
    return fmap.numpy()[0], tape.gradient(score, fmap).numpy()[0]


def verify(model, body, head_layers, tflite_bytes, ops, weights, inputs):
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]
    head = Head(ops, weights)
    keras_probs = model.predict(inputs, verbose=0)
    if not _ends_with_softmax(head_layers):
        keras_probs = np.stack([_softmax(z) for z in keras_probs])
    worst_prob, agree, worst_grad = 0.0, 0, 0.0
    for x, ref in zip(inputs, keras_probs):
        interp.set_tensor(inp, x[None].astype(np.float32))
        interp.invoke()
        logits, _ = head.forward(interp.get_tensor(out)[0])
        probs = _softmax(logits)
        worst_prob = max(worst_prob, float(np.abs(probs - ref).max()))
        agree += int(probs.argmax() == ref.argmax())
        # Gradient check on the float body, so only the NumPy head is being tested.
        cls = int(ref.argmax())
        fmap, ref_grad = tf_logit_grad(body, head_layers, x[None], cls)
        _, cache = head.forward(fmap)
        one_hot = np.zeros(len(ref), np.float32)
        one_hot[cls] = 1
        grad = head.backward(one_hot, cache)
        scale = np.abs(ref_grad).max() + 1e-12
        worst_grad = max(worst_grad, float(np.abs(grad - ref_grad).max() / scale))
    print(f"verify: top-1 agreement {agree}/{len(inputs)}, max |prob diff| {worst_prob:.4f}, "
          f"max relative grad error {worst_grad:.2e}")
    if worst_grad > 1e-3:
        raise RuntimeError("NumPy head gradients do not match TensorFlow")
    if agree < len(inputs) * 0.97 or worst_prob > 0.1:
        print("WARNING: the exported model differs noticeably from the Keras model; "
              "consider --quantize float16 or none.")
    return {"top1_agreement": agree / len(inputs), "max_prob_diff": worst_prob}


def load_verify_inputs(model, verify_dir, input_scale, resize, n_random=16, images=None):
    from inference import open_image, resize_for_model

    size = model.input_shape[1]
    rng = np.random.default_rng(0)
    xs = [rng.uniform(0, 255, (size, size, 3)) for _ in range(n_random)]
    if images is not None:
        xs.extend(np.asarray(images, np.float32))
    if verify_dir:
        for root, _, files in os.walk(verify_dir):
            for f in sorted(files):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    with open(os.path.join(root, f), "rb") as fh:
                        try:
                            img = open_image(fh.read())
                        except ValueError:
                            continue
                    xs.append(np.asarray(resize_for_model(img, size, resize), np.float32))
    return np.stack(xs).astype(np.float32) * input_scale


def export(model, out_dir, name, labels, input_scale, resize, quantize="int8", temperature=1.0,
           confidence_threshold=0.7, metrics=None, description="", feature_layer=None, verify_dir=None,
           verify_images=None):
    body, head_layers = split_model(model, feature_layer)
    ops, weights = head_to_ops(head_layers)
    tflite_bytes = convert_body(body, quantize)
    inputs = load_verify_inputs(model, verify_dir, input_scale, resize, images=verify_images)
    check = verify(model, body, head_layers, tflite_bytes, ops, weights, inputs)

    os.makedirs(out_dir, exist_ok=True)
    body_file, head_file = f"{name}_body.tflite", f"{name}_head.npz"
    with open(os.path.join(out_dir, body_file), "wb") as f:
        f.write(tflite_bytes)
    np.savez_compressed(os.path.join(out_dir, head_file), **weights)
    meta = {
        "name": name,
        "description": description,
        "body": body_file,
        "head_weights": head_file,
        "head_ops": ops,
        "input_size": int(model.input_shape[1]),
        "input_scale": float(input_scale),
        "resize": resize,
        "labels": list(labels),
        "display_names": [DISPLAY_NAMES.get(l, l.title()) for l in labels],
        "temperature": float(temperature),
        "confidence_threshold": float(confidence_threshold),
        "quantization": quantize,
        "export_check": check,
        "metrics": metrics or {},
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported {name} to {out_dir}: body {len(tflite_bytes) / 1e6:.1f} MB")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Keras .h5/.keras model")
    p.add_argument("--out-dir", default=os.path.join(ROOT, "models"))
    p.add_argument("--name", default="model")
    p.add_argument("--labels", nargs="+", required=True, help="class names in the model's output order")
    p.add_argument("--input-scale", type=float, default=1.0, help="multiplier applied to 0-255 pixels")
    p.add_argument("--resize", choices=["nearest", "bilinear", "bicubic"], default="bilinear")
    p.add_argument("--quantize", choices=["int8", "float16", "none"], default="int8")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--confidence-threshold", type=float, default=0.7)
    p.add_argument("--feature-layer", help="backbone layer to use as the Grad-CAM feature map")
    p.add_argument("--metrics-json", help="JSON file with evaluation metrics to embed in the metadata")
    p.add_argument("--description", default="")
    p.add_argument("--verify-dir", help="optional folder of real images used to verify the export")
    a = p.parse_args()

    model = tf.keras.models.load_model(a.model, compile=False)
    metrics = None
    if a.metrics_json:
        with open(a.metrics_json) as f:
            metrics = json.load(f)
    export(model, a.out_dir, a.name, a.labels, a.input_scale, a.resize, a.quantize, a.temperature,
           a.confidence_threshold, metrics, a.description, a.feature_layer, a.verify_dir)


if __name__ == "__main__":
    main()
