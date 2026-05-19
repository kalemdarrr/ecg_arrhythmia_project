# ECG Arrhythmia Classification with Deep Learning

This repository contains a PyTorch implementation for classifying ECG arrhythmias using the MIT-BIH Arrhythmia Database from PhysioNet. The project demonstrates a multi-layer deep learning architecture and includes an ablation study to analyze the impact of different components.

## Project Architecture

The complete model architecture (`ECGModel`) consists of 5 distinct blocks:
1. **Denoising Autoencoder (AE)**: Used for feature extraction and noise reduction.
2. **1D Convolutional Neural Network (1D CNN)**: Extracts local temporal features (e.g., QRS complexes).
3. **Bidirectional GRU (BiGRU)**: Captures sequential dependencies across the heartbeat.
4. **Attention Mechanism**: Focuses on the most critical temporal features.
5. **Dense Classifier**: Fully connected layers for the final 5-class AAMI standard classification.

## Ablation Study

We evaluate the necessity of each component through the following experiments:
- **Experiment 1**: CNN Only
- **Experiment 2**: CNN + BiGRU
- **Experiment 3**: AE + CNN + BiGRU
- **Experiment 4**: AE + CNN + BiGRU + Attention

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ecg-arrhythmia-project.git
cd ecg-arrhythmia-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the training and ablation study script:
```bash
python train.py
```
*Note: The script automatically downloads the MIT-BIH database via `wfdb`.*

## Google Colab

For full dataset training, it is highly recommended to run this project on Google Colab with a GPU. 
Upload the `.py` files and `requirements.txt` to your Colab environment or clone your GitHub repository directly in a Colab notebook.

## Dataset
- **Source**: MIT-BIH Arrhythmia Database (PhysioNet)
- **Classes**: N (Normal), S (Supraventricular), V (Ventricular), F (Fusion), Q (Unknown)
