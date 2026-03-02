# breast-cancer-adjuvant-therapy-predictor

# Breast Cancer Adjuvant Therapy Predictor 🧬🩺

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![SHAP](https://img.shields.io/badge/SHAP-Interpretability-green)

An interactive, AI-driven clinical decision support system that recommends personalized adjuvant therapies (Radiotherapy, Chemotherapy, Hormone Therapy) for breast cancer patients. 

This project utilizes a **Multimodal Intermediate Fusion Neural Network** to capture synergistic biological patterns between structured clinical metadata and high-dimensional genomic expression profiles (RNA-Seq) from the METABRIC dataset.

## ✨ Key Features
* **Multimodal Deep Learning:** A dual-branch architecture that independently processes 11 clinical features and 1,000 highly variant genomic features before fusing them to predict complex, multi-label treatment combinations.
* **Cost-Sensitive Learning (Threshold Tuning):** Replaces rigid 0.5 probability thresholds with a dynamic "Human-in-the-Loop" interface, allowing oncologists to calibrate recall and precision based on specific clinical safety requirements.
* **Minority Class Recovery:** Implements dynamic sample-weighting to successfully predict isolated therapies (like Chemotherapy) in highly imbalanced datasets, achieving an AUC of 0.92.
* **Clinical Interpretability:** Integrates SHAP (SHapley Additive exPlanations) to break the "black box" of deep learning. The UI provides dynamic text inferences and color-coded bar charts detailing the exact clinical and genomic factors driving every prediction.

## 🧠 Architecture Overview
The model uses an intermediate fusion strategy:
1. **Clinical Branch:** Processes structured tabular data (e.g., Nottingham Prognostic Index, Age, ER Status) utilizing Min-Max scaling and One-Hot Encoding.
2. **Genomic Branch:** Processes RNA-Seq data, utilizing log2 transformations and variance-based feature selection to isolate the top 1000 most active genes.
3. **Fusion Layer:** Concatenates the extracted representations from both modalities before passing them through dense layers optimized with dropout and batch normalization.

## 🚀 Installation and Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/darshan-1832/breast-cancer-adjuvant-therapy-predictor.git](https://github.com/darshan-1832/breast-cancer-adjuvant-therapy-predictor.git)
   cd breast-cancer-adjuvant-therapy-predictor
