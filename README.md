<div align="center">

# MRI Brain Tumor Diagnosis System

![Brain Tumor Detection Banner](https://img.shields.io/badge/AI-Medical%20Imaging-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-green?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An AI-powered deep learning system for automated brain tumor detection and classification from MRI scans**

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Render-46E3B7?style=for-the-badge)](https://mri-brain-tumor-diagnosis-system.onrender.com/)

[Features](#-key-features) • [Tech Stack](#-tech-stack--libraries) • [Installation](#️-installation) • [Usage](#-usage) • [Deployment](#️-deployment)

---

</div>

## Overview

This project harnesses the power of **deep learning** to tackle a critical medical challenge: accurately classifying brain tumors from MRI scans. Built using **transfer learning with the VGG16 architecture**, it classifies tumors into four key categories: **glioma**, **meningioma**, **pituitary**, or **no tumor**. 

The web app shows class probabilities and Grad-CAM heatmaps for every prediction, rejects images that are not MRI slices, and runs on a lightweight LiteRT runtime.

## Key Features

**Explainable predictions**
Every result shows the probability of all four classes and a **Grad-CAM heatmap** highlighting the regions that drove the prediction, with an opacity slider to compare it with the original scan.

**Safer results**
Uploads that clearly aren't MRI slices (colour photos, blank images, drawings) are rejected instead of being given a tumour label, and low-confidence predictions are reported as **inconclusive** rather than as a diagnosis.

**Lightweight, fast deployment**
The web app runs the model with **LiteRT (TensorFlow Lite)**. The convolutional body is an int8 model (~15 MB), and the small classification head runs in NumPy, which also gives exact Grad-CAM gradients. No TensorFlow is needed at runtime: the service uses ~150 MB of RAM (previously ~570 MB) and fits Render's free tier.

**Secure upload handling**
Images are processed in memory and never stored. Uploads are size-limited (10 MB), validated, and errors come back as clear messages.

**Modern interface**
Drag & drop or click to upload, a background analysis with a loading state, a printable/PDF report, optional one-click sample scans, and an "About the model" panel showing the evaluation metrics and their caveats.

**Reproducible training pipeline**
`training/train.py` retrains the model with sorted class labels, perceptual-hash **de-duplication** (removes train/test leakage), a validation split with early stopping, augmentation, EfficientNetV2/ConvNeXt backbones, clean test evaluation and **temperature-scaling calibration**. It then exports a verified model the app picks up automatically. See [training/README.md](training/README.md).

## Tech Stack & Libraries

<div align="center">

| Technology | Purpose |
|------------|---------|
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Model training, transfer learning and export |
| ![LiteRT](https://img.shields.io/badge/LiteRT-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Lightweight on-server inference |
| ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white) | Web application and JSON API |
| ![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?style=flat&logo=gunicorn&logoColor=white) | Production WSGI server |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Classification head and Grad-CAM |
| ![Pillow](https://img.shields.io/badge/Pillow-663399?style=flat) | Image decoding and preprocessing |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Evaluation metrics |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) | Training plots |

</div>

## Model Architecture

The deployed model is the VGG16 transfer-learning model trained in `CNN_brain_tumor_detection.ipynb`:

```
Input MRI image (128x128x3, pixels scaled to 0-1)
         ↓
┌─────────────────────────────┐
│ VGG16 base (ImageNet)       │  ← runs in LiteRT (int8)
│ last 3 conv layers trainable│
└─────────────────────────────┘
         ↓  block5_conv3 feature map (8x8x512) → used for Grad-CAM
┌─────────────────────────────┐
│ MaxPool → Flatten           │  ← runs in NumPy
│ Dropout (0.3)               │
│ Dense (128, ReLU)           │
│ Dropout (0.2)               │
│ Dense (4) → Softmax         │
└─────────────────────────────┘
         ↓
Glioma · Meningioma · No Tumor · Pituitary
```

Reported test accuracy is **96%** on the Kaggle test split (1,311 images). Because that dataset contains
near-duplicates across its splits, this number is probably optimistic. Retraining with `training/train.py`
reports metrics on a de-duplicated split.

## Installation

Prerequisites: Python 3.11 and pip.

```bash
git clone https://github.com/PasinduSuraweera/mri-brain-tumor-diagnosis-system.git
cd mri-brain-tumor-diagnosis-system
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` only contains what the web app needs. For training, install `training/requirements.txt`.

## Usage

**Run the web app**

```bash
python main.py                  # development server on http://localhost:5000
# or, as in production:
gunicorn main:app --workers 1 --threads 4 --timeout 120
```

**Use the API**

```bash
curl -F "file=@scan.jpg" http://localhost:5000/api/predict
```

Returns JSON with `status` (`ok`, `inconclusive`, `rejected` or `error`), the top prediction, all class
probabilities and a Grad-CAM overlay as a data URL. `GET /healthz` is a health check.

**Add sample scans**

Drop a few images named like `glioma_1.jpg` or `notumor_1.jpg` into `static/samples/` and they appear as
one-click examples on the home page (see `static/samples/README.md`).

**Retrain the model**

See [training/README.md](training/README.md). On Google Colab:

```bash
python training/train.py --data-dir "/content/drive/MyDrive/MRI Images" --export-dir models
```

## Deployment

The app is deployed on **Render**: [https://mri-brain-tumor-diagnosis-system.onrender.com/](https://mri-brain-tumor-diagnosis-system.onrender.com/)

Render settings (also captured in `render.yaml`):

| Setting | Value |
|---------|-------|
| Build command | `pip install -r requirements.txt` |
| Start command | `gunicorn main:app --workers 1 --threads 4 --timeout 120` |
| Health check path | `/healthz` |

The free tier sleeps after inactivity, so the first request after a while takes longer. A `Procfile` is
included for Heroku-style platforms.

## Project Structure

```
mri-brain-tumor-diagnosis-system/
├── main.py                       # Flask app: page, /api/predict, /healthz
├── inference.py                  # LiteRT body + NumPy head, input checks, Grad-CAM
├── templates/index.html          # Web interface
├── static/samples/               # Optional one-click sample scans
├── models/
│   ├── model.h5                  # Original Keras model (Git LFS, used for export)
│   ├── vgg16_body.tflite         # Served conv body (int8)
│   ├── vgg16_head.npz            # Served classification head weights
│   └── model_meta.json           # Labels, preprocessing, calibration, metrics
├── training/
│   ├── train.py                  # Reproducible training pipeline
│   ├── export_model.py           # Keras → web-app export with verification
│   ├── vgg16_metrics.json        # Metrics of the original notebook model
│   ├── requirements.txt          # Training dependencies
│   └── README.md
├── CNN_brain_tumor_detection.ipynb  # Original training notebook
├── requirements.txt              # Web app dependencies
├── render.yaml / Procfile        # Deployment config
└── README.md
```

## Dataset

This project works with MRI brain scan datasets organized into four categories:
- **Glioma** - A type of tumor that occurs in the brain and spinal cord
- **Meningioma** - A tumor that arises from the meninges
- **Pituitary** - A tumor in the pituitary gland
- **No Tumor** - Healthy brain scans

Recommended Datasets:
- [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Brain MRI Images for Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

**Important**: Ensure you have proper authorization and follow ethical guidelines when using medical data.

## Important Disclaimer

> **This model is not 100% accurate and is trained on a relatively small dataset**, which may impact its performance on new, unseen cases. 
>
> **This system should be viewed as a supportive diagnostic aid rather than a definitive medical tool.** It is designed for research and educational purposes and should **NOT** be used as a substitute for professional medical diagnosis. 
>
> Always consult qualified healthcare professionals for medical decisions.

## Conclusion

This MRI Brain Tumor Diagnosis System showcases how **transfer learning with VGG16** can be effectively applied to challenging medical imaging tasks, even when working with limited data. By combining strategic data augmentation, fine-tuning, and comprehensive evaluation, the system achieves promising tumor classification accuracy.

This project lays a solid foundation for future AI-driven diagnostic applications and underscores deep learning's potential to empower healthcare professionals with faster, more informed decision-making.

Future Directions
- Expanding the dataset with more diverse MRI scans
- Including additional tumor classes and subtypes
- A learned out-of-distribution detector to complement the current input checks
- Progressing toward clinical deployment and validation
- Exploring Vision Transformers and 3D (volume-level) models
- Multi-center validation studies

## Contributing

Contributions are welcome! Whether it's bug fixes, feature additions, or documentation improvements, your help is appreciated.

How to Contribute:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Pasindu Suraweera**

- GitHub: [@PasinduSuraweera](https://github.com/PasinduSuraweera)
- Live Demo: [MRI Brain Tumor Diagnosis System](https://mri-brain-tumor-diagnosis-system.onrender.com/)

## Acknowledgments

- Medical imaging research community
- TensorFlow and Keras development teams
- ImageNet and VGG16 researchers
- Open-source contributors
- Healthcare professionals providing guidance and feedback

## Contact

For questions, feedback, or collaboration opportunities, please open an issue in the repository or reach out through GitHub.

---

<div align="center">

**If you find this project helpful, please consider giving it a star!**

**Made for advancing healthcare AI**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=PasinduSuraweera.MRI-Brain-Tumor-Diagnosis-System)

*Building the future of medical diagnostics, one model at a time*

</div>
