
# 🧠 Automated Epileptic Event Classification from Low-Channel EEG

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Performance](https://img.shields.io/badge/AUROC-0.9494-success.svg)](#-final-performance-metrics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Visual EEG classification is subjective, error-prone, and time-consuming. This project introduces a robust, transfer-learning and sequence-based transformer framework (EEGformer) to automate the detection and localization of epileptiform discharges from low-channel EEG signals.**

## 🎯 Project Overview

Epilepsy affects approximately 50 million individuals globally. The WHO estimates that nearly 70% of patients could achieve freedom from seizures with appropriate diagnosis and treatment. However, significant diagnostic gaps exist—real-time, all-day monitoring by medical experts is practically unfeasible, and subjective visual analysis often leads to misdiagnosis. 

**AI-EEG in Epilepsy** offers a highly scalable, high-fidelity solution for:
* Automated Seizure Detection
* High-Fidelity Localization of Epileptogenic Foci
* Long-Term Remote Monitoring & Warning

This repository houses the complete codebase, model architectures, and Explainable AI (XAI) pipelines for classifying normal versus abnormal (epileptic) brain waves. By engineering solutions for severe dataset imbalances and evolving from baseline 1D-CNNs to advanced **EEGformer** architectures, this project achieves highly sensitive, clinically viable detection metrics.

---

## 📊 Final Performance Metrics

Our empirical testing concluded that while SMOTE+ENN was exceptional at catching abnormal signals (Recall: 93%), it sacrificed precision. By ensembling our models and optimizing the probability threshold to **0.7**, we achieved a highly stable, clinically viable system.

| Model Architecture | Accuracy | AUROC | Macro F1-Score | Abnormal Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (1D-CNN)** | 89.00% | 0.9400 | 0.69 | 0.30 |
| **SMOTE + ENN (CNN)** | 90.00% | 0.9517 | 0.84 | **0.93** |
| **Ensemble (Threshold 0.7)** | **92.56%** | **0.9494** | **0.87** | 0.71 |

---

## 🔍 Explainable AI (XAI): Peeking into the Black Box

In medical AI, trust is just as important as accuracy. We implemented **t-SNE** and **Saliency Maps** to validate *why* the model makes its decisions and to ensure it is learning true physiological markers rather than memorizing background noise.

### 1. t-SNE Clustering (Spatial Separation)
By mapping the high-dimensional probabilities into a 2D student t-probability space, we visualized the model's latent space. The clusters show clear boundaries between normal and abnormal EEGs, proving the model successfully disentangled the features.

<p align="center">
  <img src="results/xai_plots/xai_tsne_comparison.png" width="700" alt="t-SNE Clustering showing Normal vs Abnormal EEG">
</p>

### 2. Saliency Maps (Model Focus)
We extracted the gradients of the target class score with respect to the input EEG signal ($S=|\frac{\partial y_{class}}{\partial x}|$). 
* **Red/Bright regions** indicate where the model strongly relied on the waveform to make an "Abnormal" prediction. 
* **Blue/Dark regions** indicate less importance or "Normal" predictions.

<p align="center">
  <img src="results/xai_plots/saliency_true_positives/sample_19.png" width="48%" alt="True Positive Saliency">
  <img src="results/xai_plots/saliency_true_negatives/sample_0.png" width="48%" alt="True Negative Saliency">
</p>
<p align="center">
  <em>Left: True Positive (Model correctly focuses on epileptiform spikes). Right: True Negative (Model correctly identifies baseline rhythmic activity).</em>
</p>

---

## 📂 Repository Structure

The codebase is modularized to separate data preprocessing, transformer models, and explainable evaluation scripts.

.
├── UGP/                        # Core package (Data, Models, Trainers)
│   ├── config_files/           # Hyperparameter configs (Epilepsy, HAR, sleepEDF)
│   ├── dataloader/             # Data augmentations and generators
│   ├── data_preprocessing/     # EEG signal preprocessing, filtering, and compression
│   ├── models/                 # Architectures (model.py, TC.py, attention.py)
│   └── trainer/                # Model training and validation loops
├── results/                    # Consolidated output plots (DO NOT IGNORE)
│   ├── final_metric/           # ROC and Confusion Matrices for baseline
│   ├── final_metric_ensemble/  # Evaluation plots for the ensembled architectures
│   └── xai_plots/              # Saliency maps and t-SNE distributions
├── dataset_storage/            # Ignored in git: Raw and processed .pt datasets
├── ensemblemodel.py            # Script for ensemble evaluation
├── minimal_baseline_train.py   # Baseline training script
├── SmoteENN.py                 # SMOTE + ENN augmentation implementation
├── XAI.py                      # Saliency mapping generation
├── XNE-Tsme.py                 # t-SNE cluster visualization
├── main.py                     # Main execution file
└── README.md


It looks like the formatting got lost when you copied it! Here is the exact Markdown syntax for that final section.

Just copy the code block below and paste it directly into your `README.md` file:

```markdown
## ⚙️ Quick Start & Installation

**1. Clone the repository & setup environment:**
```bash
git clone [https://github.com/adianandgit/EEG-Epilepsy-Classification.git](https://github.com/adianandgit/EEG-Epilepsy-Classification.git)
cd EEG-Epilepsy-Classification
python -m venv venv
source venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt

```

**2. Download the Datasets:**

> **Note:** Due to GitHub file size limits, the massive `.pt` dataset files are hosted externally.

* Download the preprocessed datasets from [Insert Link to Google Drive / Kaggle here].
* Place them in the `dataset_storage/` directory as outlined in the file tree above.

**3. Run the Code:**

```bash
# Train the baseline model
python minimal_baseline_train.py --config UGP/config_files/Epilepsy_Configs.py

# Train using SMOTE + ENN augmentations
python SmoteENN.py

# Evaluate the ensemble network
python ensemblemodel.py 

# Generate t-SNE and Saliency Maps
python XAI.py 
python XNE-Tsme.py 

```

---

## 🔮 Future Scope

* **Cross-Subject Validation:** Testing the model's F1 score stability across different patients to guarantee true generalization.
* **Advanced Data Augmentation:** Implementing noise injection, time warping, and amplitude scaling to make the model completely robust to real-world artifacts.
* **Native Multi-Channel Architecture:** Moving away from channel compression to build a native spatial convolution transformer that processes all electrodes simultaneously.

---

## 🙏 Credits & Acknowledgments

* **Dr. Tushar Sandhan:** Project Supervisor.
* **TS-TCC Framework:** Parts of our training pipeline utilize structural groundwork from [Emadeldeen24's TS-TCC](https://github.com/emadeldeen24/TS-TCC).
* **EEGformer:** Architectural inspiration drawn from *EEGformer: A transformer-based brain activity classification method using EEG signal* (Wan et al., 2023).

```

```
