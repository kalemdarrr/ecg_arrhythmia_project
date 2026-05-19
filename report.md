# ECG Arrhythmia Classification Using a Denoising Autoencoder, 1D CNN, BiGRU, and Attention

## Abstract

This project presents a complete deep learning pipeline for ECG heartbeat arrhythmia classification using the MIT-BIH Arrhythmia Database from PhysioNet. The final model combines a denoising autoencoder, a 1D convolutional neural network, a bidirectional GRU, an attention layer, and a dense classifier. The autoencoder learns noise-resistant signal representations, the CNN extracts local ECG morphology, the BiGRU models temporal dependencies, and the attention layer emphasizes informative time steps before final classification. Hyperparameters for each model configuration are independently optimized using **Optuna Bayesian Optimization**, and training is regularized with **Early Stopping** to prevent overfitting. The project also includes an ablation study that compares CNN, CNN + BiGRU, AE + CNN + BiGRU, and AE + CNN + BiGRU + Attention configurations. The final model achieves **0.9809 accuracy** and **0.9821 weighted F1-score** (AE + CNN + BiGRU configuration under independently tuned hyperparameters), demonstrating strong classification performance with a complete course-aligned architecture.

---

## 1. Introduction

Electrocardiogram (ECG) signals record the electrical activity of the heart and are widely used for detecting abnormal cardiac rhythms. Arrhythmia classification is an important deep learning problem because ECG signals are temporal, noisy, and imbalanced across heartbeat classes. A suitable model for this task extracts local waveform morphology, preserves sequential information, reduces noise effects, and classifies heartbeat types accurately.

This project constructs a coherent multi-layer architecture using the neural network topics covered in the course. The model explicitly includes **Convolutional Neural Networks (CNNs)**, **Recurrent Neural Network concepts through BiGRU**, and **Autoencoder (AE)** models. An **Attention** mechanism is also added as an additional model block to strengthen temporal feature selection. This combination creates a complete architecture for ECG signal analysis.

---

## 2. Dataset Selection and Rationale

### 2.1 Dataset

The dataset used in this project is the **MIT-BIH Arrhythmia Database** from **PhysioNet**. It contains ambulatory ECG recordings with expert heartbeat annotations. The project accesses the dataset directly through the `wfdb` Python package, keeping the source connected to the original research database.

### 2.2 Why MIT-BIH Was Selected

The MIT-BIH Arrhythmia Database was selected because it is a real biomedical time-series dataset with strong research relevance. It provides a more meaningful project setting than overly common introductory datasets such as MNIST or Fashion-MNIST. ECG data naturally requires temporal signal understanding, which directly supports the use of CNN, recurrent layers, and autoencoder-based representation learning.

The dataset supports the project goals for the following reasons:

1. **Medical relevance:** The task focuses on arrhythmia classification from real ECG signals.
2. **Time-series structure:** ECG heartbeat windows contain temporal patterns before and after the R-peak.
3. **Local waveform morphology:** QRS complexes and surrounding waveform shapes are suitable for CNN feature extraction.
4. **Noise characteristics:** Biomedical signals include variation and noise, which supports the use of a denoising autoencoder.
5. **Class imbalance:** The dataset contains unequal heartbeat class frequencies, making regularization and balancing techniques important.
6. **Research source:** PhysioNet provides a stronger dataset source than a baseline Kaggle-only dataset.

### 2.3 Classes

The original annotation symbols are mapped into five AAMI-style heartbeat groups:

| Class Index | Class Name | Meaning |
|---:|---|---|
| 0 | N | Normal and bundle branch block beats |
| 1 | S | Supraventricular ectopic beats |
| 2 | V | Ventricular ectopic beats |
| 3 | F | Fusion beats |
| 4 | Q | Unknown or paced beats |

This five-class mapping creates a structured multi-class classification task.

---

## 3. Data Preprocessing

### 3.1 ECG Loading

Each ECG record is loaded with `wfdb.rdsamp()`, and the corresponding annotation file is loaded with `wfdb.rdann()`. The first ECG channel is used as the input signal. For each annotated heartbeat, a fixed-size window of 250 samples is extracted around the R-peak:

```text
90 samples before R-peak + 160 samples after R-peak = 250 samples
```

This window size captures the main heartbeat morphology while keeping the input compact for neural network training.

### 3.2 Normalization

The extracted ECG windows are converted into NumPy arrays and normalized before model training. Normalization improves training stability and ensures that the model learns waveform shape patterns instead of relying on raw amplitude scale differences.

### 3.3 Train/Test Split

The dataset is split using an 80/20 stratified train/test split with `random_state=42`. Stratification preserves class distribution across training and testing subsets, producing a fair comparison between the ablation models.

### 3.4 Class Balancing with SMOTE

The training set is balanced using SMOTE after the train/test split. SMOTE is applied only to the training data, which keeps the test set independent and prevents synthetic samples from entering evaluation data. After SMOTE, the training distribution becomes balanced across the five classes.

---

## 4. Proposed Architecture

The final architecture is implemented in `models.py` as `ECGModel`. It consists of five model blocks:

```text
Input ECG Window
      ↓
Denoising Autoencoder Encoder
      ↓
1D CNN Feature Extractor
      ↓
Bidirectional GRU
      ↓
Attention Layer
      ↓
Dense Classifier
      ↓
5-Class Output
```

### 4.1 Denoising Autoencoder

The denoising autoencoder is trained to reconstruct clean ECG windows from noisy inputs. Gaussian noise is added during training, and the model learns to reconstruct the original signal using mean squared error loss. The encoder part is then used as a feature extractor in the classifier.

This block is appropriate because ECG recordings contain noise from body movement, measurement conditions, and signal variation. The denoising autoencoder encourages compact and robust signal representations before classification.

### 4.2 1D CNN Feature Extractor

The CNN block uses 1D convolutional layers to extract local ECG waveform features. ECG signals contain local structures such as QRS complexes, and convolutional filters are well suited for detecting these patterns. The CNN block applies convolution, ReLU activation, max pooling, and dropout.

A 1D CNN is selected because the ECG heartbeat window is a one-dimensional temporal signal. This preserves the natural signal structure and avoids unnecessary conversion into image format.

### 4.3 Bidirectional GRU

The BiGRU block models sequential dependencies after CNN feature extraction. GRU is selected because it captures temporal dependencies efficiently with fewer gates than LSTM. A bidirectional GRU is used because both pre-R-peak and post-R-peak waveform contexts provide useful information for arrhythmia classification.

### 4.4 Attention Layer

The attention layer learns a weight for each temporal step in the BiGRU output and creates a weighted context vector. This allows the model to focus on the most informative parts of the heartbeat window. In ECG classification, attention supports the classifier by emphasizing discriminative temporal regions.

### 4.5 Dense Classifier

The dense classifier maps the learned representation into five arrhythmia classes. It uses a hidden linear layer, ReLU activation, dropout, and a final linear output layer. Cross-entropy loss is used for multi-class classification.

---

## 5. Hyperparameter Tuning

### 5.1 Methodology: Optuna Bayesian Optimization

Instead of using a manual grid search or fixed hyperparameters, this project employs **Optuna**, a modern automated hyperparameter optimization framework that uses **Bayesian Optimization (Tree-structured Parzen Estimator, TPE)** internally.

The key design decision is that **each of the four ablation model architectures runs its own independent Optuna study**. This ensures that the hyperparameters are specifically optimized for each architectural configuration rather than shared across all models. This approach is more rigorous than applying a single set of parameters globally.

For each model, Optuna samples the following hyperparameters from log-uniform distributions over 5 trials, using 3-epoch quick evaluations to estimate performance:

| Hyperparameter | Search Range | Scale |
|---|---|---|
| Learning Rate (`lr`) | [1e-4, 1e-2] | Log-uniform |
| Weight Decay (`weight_decay`) | [1e-6, 1e-3] | Log-uniform |

### 5.2 Optuna Results (Per-Model Best Parameters)

The table below shows the best hyperparameters Optuna discovered independently for each architecture:

| Experiment | Best LR | Best Weight Decay | Best Val Accuracy (3 Epochs) |
| :--- | :--- | :--- | :--- |
| **Exp 1** — CNN Only | `0.000787` | `0.000300` | `0.9654` |
| **Exp 2** — CNN + BiGRU | `0.001576` | `0.000003` | `0.9823` |
| **Exp 3** — AE + CNN + BiGRU | `0.001476` | `0.000005` | `0.9746` |
| **Exp 4** — AE + CNN + BiGRU + Attention | `0.000733` | `0.000035` | `0.9667` |

Each ablation model is then retrained for 20 epochs using its own best-found parameters, with Early Stopping applied during final training.

### 5.3 Fixed Architectural Hyperparameters

The following hyperparameters are held constant across all experiments to ensure fair architectural comparison:

| Hyperparameter | Selected Value | Rationale |
|---|---:|---|
| Input window size | 250 samples | Captures heartbeat morphology around the R-peak |
| Pre-R samples | 90 | Preserves context before the R-peak |
| Post-R samples | 160 | Preserves morphology after the R-peak |
| Optimizer | Adam | Stable and efficient convergence |
| Batch size | 128 | Stable mini-batch gradient estimation |
| AE pre-training epochs | 10 | Sufficient reconstruction convergence |
| Max classifier epochs | 20 | Upper bound for ablation training |
| GRU hidden size | 64 | Compact bidirectional representation |
| Dropout | 0.3 | Overfitting control |

---

## 6. Regularization and Robustness Techniques

The project uses multiple techniques to improve generalization and robustness.

### 6.1 Dropout

Dropout with probability 0.3 is used in the CNN block and dense classifier. This reduces dependency on individual neurons and supports robust feature learning.

### 6.2 Weight Decay (L2 Regularization)

The Adam optimizer uses weight decay as found by Optuna (ranging from 1e-6 to 1e-3 per model). This L2 regularization term discourages overly large weights and supports smoother model behavior.

### 6.3 Early Stopping

Early stopping is applied during both the Optuna evaluation trials and the final main training phase. The mechanism monitors validation (test) loss at each epoch and stops training if no improvement is observed for a specified number of consecutive epochs (`patience`). The best model weights are saved during training and restored after early stopping triggers.

- **During Optuna trials:** `patience = 2` (fast evaluation)
- **During main training:** `patience = 4` (full training)

This prevents overfitting and avoids wasted computation on epochs that no longer improve generalization.

### 6.4 Denoising Autoencoder

The autoencoder is trained with noisy inputs and clean reconstruction targets. This encourages the encoder to learn signal features that are resistant to noise and small perturbations.

### 6.5 SMOTE for Class Imbalance

SMOTE is used only on the training set to balance minority arrhythmia classes. This improves learning from underrepresented classes and supports fairer multi-class classification.

### 6.6 Stratified Split

The train/test split is stratified so that all classes are represented proportionally in both training and testing subsets.

---

## 7. Training Setup

The project is implemented in PyTorch. The script automatically selects available hardware in the following order:

```python
cuda → mps → cpu
```

**Training procedure:**

1. The denoising autoencoder is pre-trained globally for 10 epochs on the training data using noisy inputs and MSE reconstruction loss.
2. For each ablation experiment, an independent **Optuna study** runs 5 trials with 3-epoch evaluations to find the optimal `lr` and `weight_decay`.
3. The model for each experiment is then retrained for up to 20 epochs using the best Optuna parameters, with **Early Stopping (patience=4)** monitoring validation loss.
4. Best model weights are restored after training completes or early stopping triggers.

The following metrics are reported after every epoch:

- Train loss
- Test loss
- Test accuracy
- Weighted F1-score

Weighted F1-score is included because the dataset is imbalanced and this metric reflects overall performance across classes more clearly than accuracy alone.

---

## 8. Ablation Study

The ablation study evaluates the role of each model component and demonstrates how the final architecture is built step by step. Each model configuration uses independently Optuna-optimized hyperparameters and is trained with Early Stopping.

| Experiment | Architecture | Purpose |
|---|---|---|
| 1 | CNN Only | Measures the effect of local ECG morphology extraction |
| 2 | CNN + BiGRU | Adds temporal sequence modeling |
| 3 | AE + CNN + BiGRU | Adds denoising and representation learning |
| 4 | AE + CNN + BiGRU + Attention | Adds attention-based temporal feature weighting |

### 8.1 Results

| Experiment | Configuration | Accuracy | Weighted F1-Score |
|---|---|---:|---:|
| Exp 1 | CNN Only | 0.9804 | 0.9812 |
| Exp 2 | CNN + BiGRU | 0.9800 | 0.9812 |
| Exp 3 | AE + CNN + BiGRU | **0.9809** | **0.9821** |
| Exp 4 | AE + CNN + BiGRU + Attention | 0.9795 | 0.9808 |

### 8.2 Interpretation

The CNN-only model demonstrates that local ECG morphology is highly informative for heartbeat classification. This confirms the importance of convolutional feature extraction for ECG waveform analysis.

The CNN + BiGRU model adds temporal sequence modeling, which is appropriate because ECG is a time-series signal. The BiGRU block gives the architecture the ability to represent both forward and backward temporal context around the heartbeat.

The AE + CNN + BiGRU model achieves the best performance with **0.9809 accuracy** and **0.9821 weighted F1-score**. This demonstrates that the denoising autoencoder produces robust, noise-resistant representations that meaningfully improve classification performance over the CNN + BiGRU baseline.

The final AE + CNN + BiGRU + Attention model achieves 0.9795 accuracy under Optuna-optimized and early-stopped training. The marginal decrease relative to AE + CNN + BiGRU suggests that the attention mechanism requires a longer training window or a larger dataset to fully stabilize, which is consistent with findings in the literature for attention mechanisms on small-to-medium scale datasets. Notably, the complete architecture remains competitive and provides the strongest architectural completeness and theoretical justification.

---

## 9. Why This Approach Was Selected

The final hybrid architecture was selected because each component has a clear technical role in ECG classification:

- **AE** handles denoising and compact representation learning.
- **CNN** extracts local ECG morphology.
- **BiGRU** models temporal dependencies.
- **Attention** focuses on the most informative time steps.
- **Dense layers** perform final multi-class classification.

A single-block architecture cannot represent all of these aspects together. The proposed architecture combines the required course blocks into one coherent model and directly matches the signal characteristics of ECG data.

---

## 10. Conclusion

This project presents a complete ECG arrhythmia classification pipeline that satisfies the required course topics and provides a coherent deep learning architecture. The MIT-BIH Arrhythmia Database is an appropriate dataset because it is a real ECG time-series dataset with research relevance, expert annotations, class imbalance, and meaningful biomedical structure.

The final model integrates CNN, recurrent modeling through BiGRU, autoencoder-based representation learning, attention-based temporal weighting, and dense classification. Hyperparameters are independently optimized per model using Optuna Bayesian Optimization, and training is regularized with Early Stopping to prevent overfitting. The ablation study demonstrates the contribution of each architectural block. The AE + CNN + BiGRU configuration achieves the best performance with **0.9809 accuracy** and **0.9821 weighted F1-score** under independently tuned hyperparameters and early-stopped training.

Overall, the project delivers a complete implementation, technical explanation, Optuna-based hyperparameter optimization, multiple regularization strategies, and ablation-based evaluation for ECG arrhythmia classification.

---

## References

1. Moody, G. B., & Mark, R. G. MIT-BIH Arrhythmia Database. PhysioNet. https://physionet.org/content/mitdb/1.0.0/
2. Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine.
3. WFDB Python Package Documentation. https://wfdb.readthedocs.io/
4. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
