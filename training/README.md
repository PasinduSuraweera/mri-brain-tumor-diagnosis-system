# Training & exporting the model

The web app serves a lightweight export of the model (a LiteRT conv body plus a NumPy head),
so it never needs TensorFlow at runtime. This folder holds everything that does.

| File | Purpose |
|------|---------|
| `train.py` | Reproducible training pipeline (de-duplication, validation split, augmentation, fine-tuning, calibration, metrics, export) |
| `export_model.py` | Converts any compatible Keras model into the web-app format and verifies it |
| `vgg16_metrics.json` | Test metrics of the original VGG16 model (from the first version of the notebook) |
| `requirements.txt` | Training dependencies (TensorFlow etc.) |
| `../CNN_brain_tumor_detection.ipynb` | One-click Colab notebook that runs the whole pipeline |

## Retrain on Google Colab (GPU)

The easiest way is the notebook
[`CNN_brain_tumor_detection.ipynb`](../CNN_brain_tumor_detection.ipynb)
([open in Colab](https://colab.research.google.com/github/PasinduSuraweera/mri-brain-tumor-diagnosis-system/blob/upgrade-app/CNN_brain_tumor_detection.ipynb)).
Select a T4 GPU runtime and choose *Run all*. It downloads the
[Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
automatically (or uses your Drive copy), trains, compares the result with the old VGG16 model, checks the
export through the web app's own inference code (including how many real scans the input checks would wrongly
reject), adds sample scans, backs everything up to Drive and can push the new model to GitHub.

Or run the script directly in a GPU runtime (dataset in Drive as `MRI Images/Training/<class>/…` and
`MRI Images/Testing/<class>/…`):

```bash
!git clone -b upgrade-app https://github.com/PasinduSuraweera/mri-brain-tumor-diagnosis-system.git   # use main after merging
%cd mri-brain-tumor-diagnosis-system
!pip install -q ai-edge-litert==2.3.0
!python training/train.py --data-dir "/content/drive/MyDrive/MRI Images" --export-dir models
```

`--export-dir models` overwrites `models/model_meta.json`, so the app switches to the new model
automatically. Commit the new `models/*` files plus `training/runs/<run>/metrics.json` if you want to keep it.
Omit `--export-dir` to write the export to `training/runs/<run>/export/` and compare first.

Useful options: `--arch efficientnetv2b2|convnext_tiny`, `--quantize int8` (smaller, less faithful), `--image-size 260`, `--finetune-epochs 40`,
`--dedup-threshold -1` (disable de-duplication), `--confidence-threshold 0.8`.

## What the pipeline does

1. **Fixed class order:** class folders are sorted and the order is saved in `model_meta.json`.
2. **De-duplication:** a 256-bit perceptual hash removes training images that duplicate a test image
   (data leakage), repeated duplicates inside the training set, and duplicates with conflicting labels.
3. **Stratified validation split** (15%) for early stopping, LR reduction and checkpointing.
4. **Augmentation:** flips, small rotations/zoom/shift, and brightness/contrast. Applied to training data only.
5. **Two-stage training:** classification head first, then the top 30% of the backbone at a low learning
   rate, with BatchNorm frozen.
6. **Clean evaluation** on the (non-augmented) test set: accuracy, per-class precision/recall/F1,
   macro AUC and confusion matrix.
7. **Calibration:** temperature scaling fit on the validation split, so a "90%" score really means
   about 90% accuracy. Expected calibration error (ECE) is reported before and after.
8. **Export** to the web-app format (float16 LiteRT by default), verified against the Keras model (predictions and Grad-CAM gradients).

Plots (`training_curves.png`, `confusion_matrix.png`) and `metrics.json` are written to the run folder.

## Re-export an existing model

```bash
python training/export_model.py --model models/model.h5 --name vgg16 \
    --labels glioma meningioma notumor pituitary --input-scale 0.00392156862745098 \
    --resize nearest --metrics-json training/vgg16_metrics.json
```
