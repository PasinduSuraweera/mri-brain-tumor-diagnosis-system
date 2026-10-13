"""Retrain the MRI brain-tumour classifier with a cleaner, reproducible pipeline.

Improvements over the original notebook:
  * class order is sorted and saved with the model (no reliance on os.listdir order)
  * near-duplicate images are removed, including train images that leak into the test set
  * a stratified validation split drives early stopping and LR scheduling
  * modern backbone (EfficientNetV2 / ConvNeXt) at 224px with flip/rotate/zoom/contrast augmentation
  * two-stage training: frozen backbone, then fine-tuning of the top of the backbone
  * the test set is evaluated *without* augmentation
  * temperature scaling calibrates the confidence scores (fit on the validation split)
  * metrics, plots and a ready-to-serve LiteRT export are written for the web app

Expected dataset layout (e.g. Kaggle "Brain Tumor MRI Dataset"):
    DATA_DIR/Training/<class>/*.jpg
    DATA_DIR/Testing/<class>/*.jpg

Example:
    pip install -r training/requirements.txt
    python training/train.py --data-dir "/content/drive/MyDrive/MRI Images" --export-dir models
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

from inference import open_image, resize_for_model  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Data loading and de-duplication
# ---------------------------------------------------------------------------
def list_split(split_dir, classes=None):
    found = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))
    classes = classes or found
    if sorted(found) != sorted(classes):
        raise ValueError(f"Class folders differ in {split_dir}: {found} vs {classes}")
    paths, labels = [], []
    for idx, cls in enumerate(classes):
        folder = os.path.join(split_dir, cls)
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(folder, name))
                labels.append(idx)
    return paths, np.array(labels), classes


def dhash(img, size=16):
    """256-bit difference hash: robust to resizing/re-compression, but specific enough
    that different MRI slices are not mistaken for duplicates."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def load_images(paths, size):
    images, hashes, keep = [], [], []
    for i, path in enumerate(paths):
        try:
            with open(path, "rb") as f:
                img = open_image(f.read())
        except ValueError:
            print(f"  skipping unreadable image {path}")
            continue
        images.append(np.asarray(resize_for_model(img, size, "bilinear"), dtype=np.uint8))
        hashes.append(dhash(img))
        keep.append(i)
    return np.stack(images), np.stack(hashes), np.array(keep)


def hamming_neighbours(query, ref, threshold, chunk=512):
    """For each query hash, a boolean mask of ref hashes within `threshold` bits."""
    for start in range(0, len(query), chunk):
        dist = POPCOUNT[query[start:start + chunk, None, :] ^ ref[None, :, :]].sum(axis=-1)
        yield start, dist <= threshold


def deduplicate(train_hashes, train_labels, test_hashes, threshold):
    """Return a keep-mask for training images plus a report.

    Drops (1) train images that are near-duplicates of any test image (leakage),
    (2) repeated near-duplicates within the training set and (3) duplicate groups
    whose copies carry different labels.
    """
    n = len(train_hashes)
    keep = np.ones(n, dtype=bool)
    leaked = np.zeros(n, dtype=bool)
    for start, near in hamming_neighbours(train_hashes, test_hashes, threshold):
        leaked[start:start + len(near)] = near.any(axis=1)
    keep &= ~leaked

    duplicates = conflicts = 0
    conflicted = np.zeros(n, dtype=bool)
    for start, near in hamming_neighbours(train_hashes, train_hashes, threshold):
        for row, mask in enumerate(near):
            i = start + row
            earlier = np.flatnonzero(mask[:i] & keep[:i])
            if len(earlier) and keep[i]:
                keep[i] = False
                duplicates += 1
                if np.any(train_labels[earlier] != train_labels[i]):
                    conflicted[earlier] = True
                    conflicts += 1
    keep &= ~conflicted
    report = {
        "train_images": int(n),
        "removed_test_leakage": int(leaked.sum()),
        "removed_train_duplicates": int(duplicates),
        "removed_conflicting_labels": int(conflicted.sum()),
        "kept": int(keep.sum()),
        "hamming_threshold": threshold,
    }
    return keep, report


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(arch, size, n_classes, weights, dropout):
    import tensorflow as tf

    apps = tf.keras.applications
    backbones = {
        "efficientnetv2b0": apps.EfficientNetV2B0,
        "efficientnetv2b2": apps.EfficientNetV2B2,
        "convnext_tiny": apps.ConvNeXtTiny,
    }
    # These backbones include their own preprocessing and expect raw 0-255 pixels.
    base = backbones[arch](include_top=False, weights=weights, input_shape=(size, size, 3))
    inputs = tf.keras.Input((size, size, 3))
    x = base(inputs, training=False)  # keep BatchNorm statistics frozen while fine-tuning
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    logits = tf.keras.layers.Dense(n_classes, name="logits")(x)
    return tf.keras.Model(inputs, logits), base


def make_dataset(x, y, batch_size, augment=None, shuffle=False, seed=0):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if augment is not None:
        ds = ds.map(lambda a, b: (augment(tf.cast(a, tf.float32), training=True), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b))
    return ds.prefetch(tf.data.AUTOTUNE)


def augmentation(seed):
    import tensorflow as tf

    L = tf.keras.layers
    return tf.keras.Sequential([
        L.RandomFlip("horizontal", seed=seed),
        L.RandomRotation(0.05, fill_mode="constant", seed=seed),
        L.RandomZoom(0.1, fill_mode="constant", seed=seed),
        L.RandomTranslation(0.05, 0.05, fill_mode="constant", seed=seed),
        L.RandomContrast(0.15, seed=seed),
        L.RandomBrightness(0.1, value_range=(0, 255), seed=seed),
    ], name="augmentation")


# ---------------------------------------------------------------------------
# Calibration and metrics
# ---------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, y, t):
    p = softmax(logits / t)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))


def fit_temperature(logits, y):
    grid = np.exp(np.linspace(np.log(0.25), np.log(10), 400))
    return float(grid[np.argmin([nll(logits, y, t) for t in grid])])


def expected_calibration_error(probs, y, bins=15):
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def evaluate(logits, y, classes, temperature, threshold):
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    probs = softmax(logits / temperature)
    pred = probs.argmax(axis=1)
    report = classification_report(y, pred, target_names=classes, output_dict=True, zero_division=0)
    confident = probs.max(axis=1) >= threshold
    try:
        auc = float(roc_auc_score(y, probs, multi_class="ovr"))
    except ValueError:
        auc = None
    return {
        "test_accuracy": float((pred == y).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "macro_auc_ovr": auc,
        "per_class": {c: {"precision": round(report[c]["precision"], 4), "recall": round(report[c]["recall"], 4),
                          "f1": round(report[c]["f1-score"], 4), "support": int(report[c]["support"])}
                      for c in classes},
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(classes))).tolist(),
        "ece_uncalibrated": expected_calibration_error(softmax(logits), y),
        "ece_calibrated": expected_calibration_error(probs, y),
        "confidence_threshold": threshold,
        "coverage_at_threshold": float(confident.mean()),
        "accuracy_at_threshold": float((pred[confident] == y[confident]).mean()) if confident.any() else None,
    }


def save_plots(run_dir, histories, metrics, classes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = {k: sum((h.get(k, []) for h in histories), []) for k in ("loss", "val_loss", "accuracy", "val_accuracy")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(hist[key], label="train")
        ax.plot(hist["val_" + key], label="validation")
        ax.axvline(len(histories[0].get("loss", [])) - 0.5, color="gray", ls="--", lw=1)
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "training_curves.png"), dpi=120)

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30)
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test confusion matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=120)
    plt.close("all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder containing Training/ and Testing/")
    p.add_argument("--arch", choices=["efficientnetv2b0", "efficientnetv2b2", "convnext_tiny"],
                   default="efficientnetv2b0")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=8, help="epochs with the backbone frozen")
    p.add_argument("--finetune-epochs", type=int, default=30, help="max epochs while fine-tuning")
    p.add_argument("--finetune-fraction", type=float, default=0.3,
                   help="fraction of backbone layers (from the top) unfrozen for fine-tuning")
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--dedup-threshold", type=int, default=6,
                   help="max differing bits (of 256) for two images to count as duplicates; -1 disables")
    p.add_argument("--confidence-threshold", type=float, default=0.7)
    p.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (for quick smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-dir", default=None, help="where checkpoints, metrics and plots go")
    p.add_argument("--export-dir", default=None,
                   help="where to write the web-app model files (default: <run-dir>/export; use 'models' to deploy)")
    a = p.parse_args()

    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    tf.keras.utils.set_random_seed(a.seed)
    run_dir = a.run_dir or os.path.join(ROOT, "runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + a.arch)
    os.makedirs(run_dir, exist_ok=True)
    export_dir = a.export_dir or os.path.join(run_dir, "export")

    # --- data -------------------------------------------------------------------------
    train_paths, train_y, classes = list_split(os.path.join(a.data_dir, "Training"))
    test_paths, test_y, _ = list_split(os.path.join(a.data_dir, "Testing"), classes)
    print(f"Classes (sorted, saved with the model): {classes}")
    print(f"Loading {len(train_paths)} training and {len(test_paths)} test images at {a.image_size}px ...")
    train_x, train_h, kept = load_images(train_paths, a.image_size)
    train_y = train_y[kept]
    test_x, test_h, kept = load_images(test_paths, a.image_size)
    test_y = test_y[kept]

    if a.dedup_threshold >= 0:
        keep, dedup = deduplicate(train_h, train_y, test_h, a.dedup_threshold)
        train_x, train_y = train_x[keep], train_y[keep]
    else:
        dedup = {"disabled": True}
    print("De-duplication:", json.dumps(dedup))

    tr_idx, va_idx = train_test_split(np.arange(len(train_y)), test_size=a.val_split,
                                      stratify=train_y, random_state=a.seed)
    x_tr, y_tr, x_va, y_va = train_x[tr_idx], train_y[tr_idx], train_x[va_idx], train_y[va_idx]
    counts = np.bincount(y_tr, minlength=len(classes))
    class_weight = {i: float(len(y_tr) / (len(classes) * c)) for i, c in enumerate(counts) if c}
    print(f"Split: train {len(y_tr)}, validation {len(y_va)}, test {len(test_y)}; class counts {counts.tolist()}")

    train_ds = make_dataset(x_tr, y_tr, a.batch_size, augmentation(a.seed), shuffle=True, seed=a.seed)
    val_ds = make_dataset(x_va, y_va, a.batch_size)
    test_ds = make_dataset(test_x, test_y, a.batch_size)

    # --- model ------------------------------------------------------------------------
    weights = None if a.weights == "none" else a.weights
    model, base = build_model(a.arch, a.image_size, len(classes), weights, a.dropout)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    ckpt = os.path.join(run_dir, "best.keras")

    def callbacks(patience):
        return [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(1, patience // 2)),
            tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "history.csv"), append=True),
        ]

    print("Stage 1: training the classification head (backbone frozen)")
    base.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=a.head_epochs, class_weight=class_weight,
                   callbacks=callbacks(3), verbose=2)

    print("Stage 2: fine-tuning the top of the backbone")
    base.trainable = True
    n_frozen = int(len(base.layers) * (1 - a.finetune_fraction))
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= n_frozen and not isinstance(layer, tf.keras.layers.BatchNormalization)
    model.compile(tf.keras.optimizers.Adam(1e-4), loss=loss, metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=a.finetune_epochs, class_weight=class_weight,
                   callbacks=callbacks(5), verbose=2)
    model.save(ckpt)

    # --- calibration + evaluation -----------------------------------------------------
    val_logits = model.predict(val_ds, verbose=0)
    temperature = fit_temperature(val_logits, y_va)
    print(f"Temperature scaling: T = {temperature:.3f} (val NLL {nll(val_logits, y_va, 1.0):.4f} -> "
          f"{nll(val_logits, y_va, temperature):.4f})")
    metrics = evaluate(model.predict(test_ds, verbose=0), test_y, classes, temperature, a.confidence_threshold)
    metrics.update({
        "source": f"training/train.py ({a.arch}, {a.image_size}px), de-duplicated test split ({len(test_y)} images)",
        "temperature": temperature,
        "deduplication": dedup,
        "split": {"train": int(len(y_tr)), "validation": int(len(y_va)), "test": int(len(test_y))},
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_plots(run_dir, [h1.history, h2.history], metrics, classes)
    print(json.dumps({k: metrics[k] for k in ("test_accuracy", "macro_f1", "macro_auc_ovr",
                                              "ece_uncalibrated", "ece_calibrated")}, indent=2))

    # --- export for the web app -------------------------------------------------------
    from export_model import export

    rng = np.random.default_rng(a.seed)
    sample = test_x[rng.choice(len(test_x), size=min(32, len(test_x)), replace=False)]
    export(model, export_dir, a.arch, classes, input_scale=1.0, resize="bilinear", temperature=temperature,
           confidence_threshold=a.confidence_threshold, metrics=metrics,
           description=f"{a.arch} ({a.image_size}x{a.image_size} input) trained with training/train.py",
           verify_images=sample)
    print(f"Run artefacts: {run_dir}\nWeb-app model files: {export_dir}")


if __name__ == "__main__":
    main()
