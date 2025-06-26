<p align="center">
  <img src="assets/Pneumonia_X_ray_Classifier_Banner.png" alt="Pneumonia Detection with X-rays" width="100%">
</p>

<h1 align="center"> Pneumonia X-ray Classifier</h1>
<p align="center"><i>From pixel to prognosis: Deep learning-powered pneumonia detection with full visual explainability</i></p>

---

## 🧠 Overview

`pneumonia-xray-classifier` is a modular machine learning pipeline that leverages the PneumoniaMNIST dataset to detect pneumonia from chest X-ray images. This project integrates deep learning, model explainability, and deployment-ready architecture—tailored for real-world medical AI use cases.

Built from the ground up with reproducibility and extensibility in mind, it reflects best practices in ML engineering—from clean code organization to inference APIs and diagnostic visualization.

---

## ⚙️ Key Features

- 🩻 Loads and preprocesses medical images via the MedMNIST PyTorch API
- 🧠 Trains CNN models for binary classification (pneumonia vs. normal)
- 🔍 Visualizes predictions using Grad-CAM heatmaps and misclassification grids
- 🔁 Includes stratified cross-validation and evaluation pipelines
- 🧪 Reports confusion matrix and test-set accuracy
- 📤 Optimized for deployment (Flask/FastAPI-ready)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Dartayous/pneumonia-xray-classifier.git
cd pneumonia-xray-classifier

🏥 Clinical Relevance
AI-driven diagnostic tools like this pneumonia classifier have the potential to assist radiologists and healthcare professionals by highlighting regions of interest and reducing diagnostic oversight. While not a replacement for medical judgment, such models can support triage in high-volume settings, offer second opinions in under-resourced clinics, and enable earlier intervention by flagging high-risk cases based on imaging alone.

The integration of Grad-CAM visualizations enhances trust and transparency by showing exactly where the model is focusing—making it a valuable aid in human-in-the-loop clinical workflows.
