# ECG Arrhythmia Classification with AE · 1D CNN · BiGRU · Attention

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/Optuna-4.8.0-blueviolet?logo=optuna" />
  <img src="https://img.shields.io/badge/Dataset-MIT--BIH%20PhysioNet-green" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

A complete deep learning pipeline for **5-class ECG heartbeat arrhythmia classification** using the **MIT-BIH Arrhythmia Database** (PhysioNet). The project directly integrates the required course blocks — **Convolutional Neural Networks**, **Bidirectional GRU**, and **Denoising Autoencoder** — extended with an **Attention mechanism**, independent **Bayesian hyperparameter optimization (Optuna)**, and **Early Stopping** regularization.

---

## Table of Contents
- [Project Summary](#1-project-summary)
- [Dataset Rationale](#2-dataset-rationale)
- [Model Architecture](#3-model-architecture)
- [Hyperparameter Tuning — Optuna](#4-hyperparameter-tuning--optuna)
- [Ablation Study Results](#5-ablation-study-results)
- [Regularization Techniques](#6-regularization-techniques)
- [Repository Structure](#7-repository-structure)
- [Installation & Usage](#8-installation--usage)
- [Project Requirement Compliance](#9-project-requirement-compliance)
- [References](#10-references)

---

## 1. Project Summary

| Item | Description |
| :--- | :--- |
| Task | 5-class ECG heartbeat arrhythmia classification |
| Dataset | MIT-BIH Arrhythmia Database (PhysioNet) |
| Input | 250-sample 1D ECG windows centered at annotated R-peaks |
| Classes | AAMI standard groups: N · S · V · F · Q |
| Framework | PyTorch 2.10 · CUDA |
| Hyperparameter Optimization | Optuna 4.8 (Bayesian Optimization / TPE) |
| Final Model | Denoising AE + 1D CNN + BiGRU + Attention + Dense Classifier |
| Best Result | **Acc: 0.9809 · F1: 0.9821** (AE + CNN + BiGRU, independently tuned) |
| Evaluation Metrics | Accuracy · Weighted F1-Score |
| Additional Analysis | 4-model ablation study with per-model Optuna tuning |

Full paper-style explanation → [`report.md`](report.md)  
Full training logs with Optuna trials and epoch tables → [`training_log.md`](training_log.md)  
Auto-generated results summary → [`training_results.md`](training_results.md)

---

## 2. Dataset Rationale

The **MIT-BIH Arrhythmia Database** (PhysioNet) was selected over introductory datasets such as MNIST because ECG is a real biomedical time-series signal with domain-specific complexity:

| Reason | Architectural Justification |
| :--- | :--- |
| 1D temporal structure | Directly supports 1D CNN and BiGRU/RNN layers |
| Local waveform morphology (QRS, P, T waves) | Suited for 1D convolutional feature extraction |
| Signal noise and variation | Motivates the use of a Denoising Autoencoder |
| Severe class imbalance (N >> S, V, F, Q) | Requires SMOTE and weighted evaluation (F1) |
| Research-grade source (PhysioNet) | Provides credibility over Kaggle-only datasets |

The original annotation symbols are mapped to five **AAMI-standard** heartbeat classes:

| Index | Class | Description |
| :---: | :---: | :--- |
| 0 | N | Normal and bundle branch block beats |
| 1 | S | Supraventricular ectopic beats |
| 2 | V | Ventricular ectopic beats |
| 3 | F | Fusion beats |
| 4 | Q | Unknown / paced beats |

---

## 3. Model Architecture

```
Input ECG Window  (Batch, 1, 250)
        │
        ▼
┌─────────────────────────────┐
│  Denoising Autoencoder (AE) │  ← Noise-resistant feature extraction
│  Conv1d Encoder: 250 → 62   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  1D CNN Feature Extractor   │  ← Local morphology (QRS, etc.)
│  Conv1d × 2 + MaxPool + Drop│
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Bidirectional GRU (BiGRU)  │  ← Forward & backward temporal context
│  hidden_size = 64 × 2 = 128 │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Attention Layer            │  ← Weighted context vector over time steps
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Dense Classifier           │  ← Linear → ReLU → Dropout → Linear
│  Output: 5 classes          │
└─────────────────────────────┘
```

The model is configurable for ablation via `use_ae`, `use_gru`, and `use_attention` flags in [`models.py`](models.py).

---

## 4. Hyperparameter Tuning — Optuna

Instead of applying fixed or grid-searched hyperparameters, each of the four ablation model architectures runs its **own independent Optuna study** using **Bayesian Optimization (TPE sampler)**. This ensures that every configuration is evaluated under its own optimal training conditions.

**Search Space (per model, log-uniform):**

| Parameter | Range |
| :--- | :--- |
| Learning Rate | [1e-4, 1e-2] |
| Weight Decay | [1e-6, 1e-3] |

**Optuna Results (5 Trials × 3 Epochs each):**

| Experiment | Best LR | Best Weight Decay | Best Val Acc (3 Epochs) |
| :--- | :--- | :--- | ---: |
| Exp 1 — CNN Only | `0.000787` | `0.000300` | 0.9654 |
| Exp 2 — CNN + BiGRU | `0.001576` | `0.000003` | 0.9823 |
| Exp 3 — AE + CNN + BiGRU | `0.001476` | `0.000005` | 0.9746 |
| Exp 4 — AE + CNN + BiGRU + Attention | `0.000733` | `0.000035` | 0.9667 |

Each model is then retrained for up to **20 epochs** using its best-found parameters with **Early Stopping (patience = 4)**.

---

## 5. Ablation Study Results

| Experiment | Configuration | Best Epoch | Accuracy | Weighted F1 |
| :--- | :--- | :---: | ---: | ---: |
| Exp 1 | CNN Only | 15 | 0.9804 | 0.9812 |
| Exp 2 | CNN + BiGRU | 3 | 0.9800 | 0.9812 |
| **Exp 3** | **AE + CNN + BiGRU** | **2** | **0.9809** | **0.9821** |
| Exp 4 | AE + CNN + BiGRU + Attention | 9 | 0.9795 | 0.9808 |

> The **AE + CNN + BiGRU** configuration achieved the best result, confirming the value of the denoising autoencoder as a robust feature extractor. Early Stopping consistently prevented overfitting across all four experiments.

---

## 6. Regularization Techniques

| Technique | Details |
| :--- | :--- |
| **Dropout** | p = 0.3 applied in CNN block and Dense Classifier |
| **L2 Weight Decay** | Individually tuned per model via Optuna (1e-6 to 1e-3) |
| **Early Stopping** | patience = 4 (main training), patience = 2 (Optuna trials) |
| **Denoising Autoencoder** | Trained with Gaussian noise (σ=0.1) → MSE reconstruction loss |
| **SMOTE** | Applied only to training data to balance minority classes |
| **Stratified Split** | 80/20 split preserving class ratios in train and test |

---

## 7. Repository Structure

```
ecg_arrhythmia_project/
├── models.py               # AE, CNN, BiGRU, Attention, ECGModel definitions
├── data_loader.py          # Dataset download, windowing, normalization, SMOTE
├── train.py                # Optuna tuning + Early Stopping + Ablation training loop
├── requirements.txt        # Python dependencies
├── report.md               # Full paper-style technical report
├── training_log.md         # Complete Optuna trial logs + epoch-by-epoch results
├── training_results.md     # Auto-generated ablation summary table
├── .gitignore
└── README.md
```

---

## 8. Installation & Usage

**Clone the repository:**
```bash
git clone https://github.com/kalemdarrr/ecg_arrhythmia_project.git
cd ecg_arrhythmia_project
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the full pipeline** (data download → Optuna tuning → ablation training → report generation):
```bash
python train.py
```

> The script automatically downloads the MIT-BIH Arrhythmia Database via `wfdb`. Running on a **GPU (CUDA)** is strongly recommended. Google Colab with a T4 GPU is a suitable free environment.

**Google Colab usage:**
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/deep/ecg_arrhythmia_project
!pip install -r requirements.txt
!python train.py
```

**View the auto-generated results report:**
```python
from IPython.display import Markdown, display
display(Markdown("training_results.md"))
```

---

## 9. Project Requirement Compliance

| Requirement | Implementation |
| :--- | :--- |
| CNN block | 1D convolutional feature extraction (`CNNFeatureExtractor` in `models.py`) |
| RNN / GRU block | Bidirectional GRU (`BiGRULayer` in `models.py`) |
| Autoencoder block | Denoising Autoencoder with noise augmentation (`DenoisingAutoencoder`) |
| Additional block | Temporal Attention Mechanism (`AttentionLayer`) |
| Hyperparameter tuning | Independent Optuna Bayesian Optimization per ablation model |
| Regularization | Dropout · L2 Weight Decay · Early Stopping · Denoising AE · SMOTE |
| Ablation study | 4-model comparison with per-model optimal hyperparameters |
| Dataset rationale | MIT-BIH PhysioNet — justified for 1D temporal + noise + imbalance characteristics |
| Paper-style explanation | Full technical report in `report.md` |
| Training logs | Complete logs with Optuna trials and epoch results in `training_log.md` |

---

## 10. References

1. Moody, G. B., & Mark, R. G. *MIT-BIH Arrhythmia Database*. PhysioNet. https://physionet.org/content/mitdb/1.0.0/
2. Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*.
3. *WFDB Python Package Documentation*. https://wfdb.readthedocs.io/
4. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.
