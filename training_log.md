# Full Training Log

**Hardware:** CUDA (Google Colab T4 GPU)  
**Framework:** PyTorch 2.10.0+cu128  
**Hyperparameter Optimization:** Optuna 4.8.0 (Bayesian Optimization, TPE)  
**Regularization:** Early Stopping (patience=4 for main training)

---

## 1. Data Preparation

| Property | Value |
| :--- | :--- |
| Total Extracted Heartbeats | 109,452 |
| Class Distribution (Before SMOTE) | N: 90,594 · S: 2,781 · V: 7,235 · F: 802 · Q: 8,040 |
| Class Distribution (After SMOTE - Training) | 72,475 per class (5 classes) |
| X_train Shape | (362,375, 1, 250) |
| y_train Shape | (362,375,) |
| X_test Shape | (21,891, 1, 250) |
| y_test Shape | (21,891,) |

---

## 2. Autoencoder Pre-training (Global — 10 Epochs)

| Epoch | Loss |
| ---: | :--- |
| 1 | 0.0044 |
| 2 | 0.0013 |
| 3 | 0.0013 |
| 4 | 0.0013 |
| 5 | 0.0013 |
| 6 | 0.0013 |
| 7 | 0.0013 |
| 8 | 0.0013 |
| 9 | 0.0013 |
| 10 | 0.0013 |

The autoencoder converged rapidly by epoch 2, indicating that the ECG signal reconstruction task was successfully learned.

---

## 3. Optuna Hyperparameter Tuning (5 Trials × 3 Epochs per Model)

### Experiment 1 — CNN Only

| Trial | LR | Weight Decay | Val Acc | Val F1 |
| ---: | :--- | :--- | ---: | ---: |
| 1 | — | — | 0.9473 | 0.9549 |
| 2 | — | — | 0.9654 | 0.9685 |
| 3 | — | — | 0.9524 | 0.9593 |
| 4 | — | — | 0.9417 | 0.9486 |
| 5 | — | — | 0.9643 | 0.9678 |

**Best Params → LR: `0.000787` · Weight Decay: `0.000300` · Val Acc: `0.9654`**

---

### Experiment 2 — CNN + BiGRU

| Trial | Val Acc | Val F1 |
| ---: | ---: | ---: |
| 1 | 0.9567 | 0.9619 |
| 2 | 0.9715 | 0.9738 |
| 3 | 0.9702 | 0.9727 |
| 4 | 0.9823 | 0.9832 |
| 5 | 0.9632 | 0.9680 |

**Best Params → LR: `0.001576` · Weight Decay: `0.000003` · Val Acc: `0.9823`**

---

### Experiment 3 — AE + CNN + BiGRU

| Trial | Val Acc | Val F1 |
| ---: | ---: | ---: |
| 1 | 0.9648 | 0.9686 |
| 2 | 0.9685 | 0.9719 |
| 3 | 0.9746 | 0.9768 |
| 4 | 0.9608 | 0.9649 |
| 5 | 0.9628 | 0.9669 |

**Best Params → LR: `0.001476` · Weight Decay: `0.000005` · Val Acc: `0.9746`**

---

### Experiment 4 — AE + CNN + BiGRU + Attention (FINAL)

| Trial | Val Acc | Val F1 |
| ---: | ---: | ---: |
| 1 | 0.9524 | 0.9593 |
| 2 | 0.9508 | 0.9579 |
| 3 | 0.9468 | 0.9547 |
| 4 | 0.9667 | 0.9700 |
| 5 | 0.9556 | 0.9609 |

**Best Params → LR: `0.000733` · Weight Decay: `0.000035` · Val Acc: `0.9667`**

---

## 4. Main Training Logs (20 Epochs Max + Early Stopping, patience=4)

### Experiment 1 — CNN Only
*Best Params: LR=0.000787, Weight Decay=0.000300*

| Epoch | Train Loss | Test Loss | Test Acc | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3631 | 0.2778 | 0.9259 | 0.9380 |
| 2 | 0.2221 | 0.1847 | 0.9486 | 0.9558 |
| 3 | 0.1820 | 0.1669 | 0.9502 | 0.9567 |
| 4 | 0.1547 | 0.2065 | 0.9255 | 0.9396 |
| 5 | 0.1335 | 0.1164 | 0.9648 | 0.9684 |
| 6 | 0.1164 | 0.1705 | 0.9414 | 0.9514 |
| 7 | 0.1046 | 0.1135 | 0.9649 | 0.9680 |
| 8 | 0.0966 | 0.0903 | 0.9735 | 0.9753 |
| 9 | 0.0906 | 0.0779 | 0.9774 | 0.9787 |
| 10 | 0.0851 | 0.0884 | 0.9728 | 0.9752 |
| 11 | 0.0806 | 0.0826 | 0.9755 | 0.9769 |
| 12 | 0.0773 | 0.0934 | 0.9700 | 0.9726 |
| 13 | 0.0745 | 0.0721 | 0.9780 | 0.9793 |
| 14 | 0.0730 | 0.0832 | 0.9757 | 0.9772 |
| 15 | 0.0715 | 0.0682 | **0.9804** | **0.9812** |
| 16 | 0.0702 | 0.0951 | 0.9709 | 0.9730 |
| 17 | 0.0699 | 0.0724 | 0.9767 | 0.9785 |
| 18 | 0.0688 | 0.0828 | 0.9762 | 0.9779 |
| 19 | 0.0677 | 0.0747 | 0.9766 | 0.9784 |

> ⚠️ **Early Stopping triggered at epoch 19** (no improvement for 4 epochs). Best weights from **Epoch 15** restored.  
> **Final: Acc = 0.9804 · F1 = 0.9812**

---

### Experiment 2 — CNN + BiGRU
*Best Params: LR=0.001576, Weight Decay=0.000003*

| Epoch | Train Loss | Test Loss | Test Acc | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1907 | 0.1001 | 0.9677 | 0.9710 |
| 2 | 0.0791 | 0.0698 | 0.9778 | 0.9792 |
| 3 | 0.0611 | 0.0692 | **0.9800** | **0.9812** |
| 4 | 0.0522 | 0.0702 | 0.9794 | 0.9807 |
| 5 | 0.0443 | 0.0779 | 0.9746 | 0.9776 |
| 6 | 0.0414 | 0.0778 | 0.9810 | 0.9823 |
| 7 | 0.0380 | 0.0710 | 0.9832 | 0.9841 |

> ⚠️ **Early Stopping triggered at epoch 7** (no improvement for 4 epochs). Best weights from **Epoch 3** restored.  
> **Final: Acc = 0.9800 · F1 = 0.9812**

---

### Experiment 3 — AE + CNN + BiGRU
*Best Params: LR=0.001476, Weight Decay=0.000005*

| Epoch | Train Loss | Test Loss | Test Acc | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1942 | 0.1157 | 0.9580 | 0.9639 |
| 2 | 0.0784 | 0.0650 | **0.9809** | **0.9821** |
| 3 | 0.0601 | 0.0816 | 0.9726 | 0.9752 |
| 4 | 0.0518 | 0.0801 | 0.9733 | 0.9759 |
| 5 | 0.0472 | 0.0805 | 0.9737 | 0.9765 |
| 6 | 0.0423 | 0.0714 | 0.9786 | 0.9802 |

> ⚠️ **Early Stopping triggered at epoch 6** (no improvement for 4 epochs). Best weights from **Epoch 2** restored.  
> **Final: Acc = 0.9809 · F1 = 0.9821**

---

### Experiment 4 — AE + CNN + BiGRU + Attention (FINAL)
*Best Params: LR=0.000733, Weight Decay=0.000035*

| Epoch | Train Loss | Test Loss | Test Acc | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3658 | 0.2313 | 0.9220 | 0.9369 |
| 2 | 0.1255 | 0.1432 | 0.9483 | 0.9559 |
| 3 | 0.0872 | 0.0817 | 0.9754 | 0.9771 |
| 4 | 0.0703 | 0.0813 | 0.9735 | 0.9757 |
| 5 | 0.0608 | 0.0863 | 0.9724 | 0.9747 |
| 6 | 0.0547 | 0.0972 | 0.9653 | 0.9704 |
| 7 | 0.0496 | 0.0731 | 0.9776 | 0.9791 |
| 8 | 0.0454 | 0.0776 | 0.9761 | 0.9779 |
| 9 | 0.0434 | 0.0703 | **0.9795** | **0.9808** |
| 10 | 0.0412 | 0.0718 | 0.9777 | 0.9792 |
| 11 | 0.0388 | 0.0884 | 0.9697 | 0.9728 |
| 12 | 0.0372 | 0.0866 | 0.9742 | 0.9765 |
| 13 | 0.0371 | 0.0796 | 0.9745 | 0.9767 |

> ⚠️ **Early Stopping triggered at epoch 13** (no improvement for 4 epochs). Best weights from **Epoch 9** restored.  
> **Final: Acc = 0.9795 · F1 = 0.9808**

---

## 5. Ablation Study — Final Results Summary

| Experiment | Configuration | Best Epoch | Accuracy | Weighted F1-Score |
| :--- | :--- | :---: | ---: | ---: |
| Exp 1 | CNN Only | 15 | 0.9804 | 0.9812 |
| Exp 2 | CNN + BiGRU | 3 | 0.9800 | 0.9812 |
| **Exp 3** | **AE + CNN + BiGRU** | **2** | **0.9809** | **0.9821** |
| Exp 4 | AE + CNN + BiGRU + Attention | 9 | 0.9795 | 0.9808 |

> **Best overall model: AE + CNN + BiGRU** (Exp 3) with `Acc = 0.9809`, `F1 = 0.9821`.  
> Early Stopping consistently prevented overfitting across all four configurations.
