# Full Training Log — Run 2

**Hardware:** CUDA (Google Colab T4 GPU)  
**Framework:** PyTorch 2.10.0+cu128  
**Hyperparameter Optimization:** Optuna 4.8.0 (Bayesian Optimization, TPE)  
**Regularization:** Early Stopping (patience=4 for main training, patience=2 for Optuna trials)  
**Note:** Confusion matrix plots were auto-saved at end of each experiment.

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
| 1 | 0.0141 |
| 2 | 0.0015 |
| 3 | 0.0015 |
| 4 | 0.0014 |
| 5 | 0.0014 |
| 6 | 0.0014 |
| 7 | 0.0013 |
| 8 | 0.0013 |
| 9 | 0.0013 |
| 10 | 0.0013 |

The autoencoder converged by epoch 2 (loss: 0.0015) and stabilized at 0.0013, consistent with Run 1 behavior.

---

## 3. Optuna Hyperparameter Tuning (5 Trials × 3 Epochs per Model)

### Experiment 1 — CNN Only

| Trial | Val Acc | Val F1 |
| ---: | ---: | ---: |
| 1 | 0.8902 | 0.9092 |
| 2 | 0.9550 | 0.9613 |
| 3 | 0.9365 | 0.9456 |
| 4 | 0.9528 | 0.9592 |
| 5 | 0.9577 | 0.9625 |

**Best Params → LR: `0.004534` · Weight Decay: `0.000372` · Val Acc: `0.9577`**

---

### Experiment 2 — CNN + BiGRU

| Trial | Val Acc | Val F1 |
| ---: | ---: | ---: |
| 1 | 0.9418 | 0.9488 |
| 2 | 0.9588 | 0.9636 |
| 3 | 0.9633 | 0.9667 |
| 4 | 0.9433 | 0.9502 |
| 5 | 0.9517 | 0.9572 |

**Best Params → LR: `0.000674` · Weight Decay: `0.000053` · Val Acc: `0.9633`**

---

### Experiment 3 — AE + CNN + BiGRU

| Trial | Val Acc | Val F1 |
| ---: | ---: | ---: |
| 1 | 0.9110 | 0.9251 |
| 2 | 0.9537 | 0.9599 |
| 3 | 0.9041 | 0.9216 |
| 4 | 0.9644 | 0.9680 |
| 5 | 0.9558 | 0.9618 |

**Best Params → LR: `0.008950` · Weight Decay: `0.000001` · Val Acc: `0.9644`**

---

### Experiment 4 — AE + CNN + BiGRU + Attention (FINAL)

| Trial | Val Acc | Val F1 |
| ---: | ---: | ---: |
| 1 | 0.9249 | 0.9375 |
| 2 | 0.7963 | 0.8399 |
| 3 | 0.9347 | 0.9442 |
| 4 | 0.9375 | 0.9465 |
| 5 | 0.8933 | 0.9140 |

**Best Params → LR: `0.001292` · Weight Decay: `0.000430` · Val Acc: `0.9375`**

---

## 4. Main Training Logs (20 Epochs Max + Early Stopping, patience=4)

### Experiment 1 — CNN Only
*Best Params: LR=0.004534, Weight Decay=0.000372*

| Epoch | Train Loss | Test Loss | Test Acc | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4085 | 0.2385 | 0.9289 | 0.9392 |
| 2 | 0.2264 | 0.1915 | 0.9415 | 0.9502 |
| 3 | 0.1925 | 0.1693 | 0.9540 | 0.9596 |
| 4 | 0.1756 | 0.1599 | 0.9500 | 0.9558 |
| 5 | 0.1630 | 0.1516 | 0.9521 | 0.9589 |
| 6 | 0.1527 | 0.1343 | 0.9586 | 0.9637 |
| 7 | 0.1439 | 0.1343 | 0.9571 | 0.9623 |
| 8 | 0.1364 | 0.1127 | 0.9667 | 0.9700 |
| 9 | 0.1317 | 0.1180 | 0.9652 | 0.9687 |
| 10 | 0.1269 | 0.1055 | 0.9677 | 0.9705 |
| 11 | 0.1226 | 0.1153 | 0.9653 | 0.9694 |
| 12 | 0.1195 | 0.1000 | 0.9709 | 0.9735 |
| 13 | 0.1151 | 0.1399 | 0.9544 | 0.9593 |
| 14 | 0.1151 | 0.1129 | 0.9669 | 0.9704 |
| 15 | 0.1107 | 0.1004 | 0.9684 | 0.9717 |
| 16 | 0.1104 | 0.0885 | 0.9731 | 0.9751 |
| 17 | 0.1086 | 0.0954 | 0.9713 | 0.9740 |
| 18 | 0.1069 | 0.1103 | 0.9662 | 0.9699 |
| 19 | 0.1055 | 0.0973 | 0.9719 | 0.9745 |
| 20 | 0.1053 | 0.0869 | **0.9747** | **0.9764** |

> ✅ Ran all 20 epochs (no early stopping triggered). Best weights from **Epoch 20** used.  
> **Final: Acc = 0.9747 · F1 = 0.9764**

---

### Experiment 2 — CNN + BiGRU
*Best Params: LR=0.000674, Weight Decay=0.000053*

| Epoch | Train Loss | Test Loss | Test Acc | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4262 | 0.2446 | 0.9308 | 0.9411 |
| 2 | 0.1646 | 0.1475 | 0.9564 | 0.9615 |
| 3 | 0.1142 | 0.1180 | 0.9666 | 0.9696 |
| 4 | 0.0898 | 0.1011 | 0.9705 | 0.9726 |
| 5 | 0.0750 | 0.1022 | 0.9664 | 0.9698 |
| 6 | 0.0649 | 0.0894 | 0.9714 | 0.9737 |
| 7 | 0.0570 | 0.0754 | 0.9778 | 0.9791 |
| 8 | 0.0536 | 0.0723 | 0.9798 | 0.9806 |
| 9 | 0.0483 | 0.0772 | 0.9765 | 0.9781 |
| 10 | 0.0446 | 0.0699 | 0.9800 | 0.9811 |
| 11 | 0.0429 | 0.0609 | **0.9836** | **0.9842** |
| 12 | 0.0414 | 0.0686 | 0.9815 | 0.9824 |
| 13 | 0.0383 | 0.0711 | 0.9797 | 0.9809 |
| 14 | 0.0373 | 0.0631 | 0.9840 | 0.9845 |
| 15 | 0.0352 | 0.0678 | 0.9798 | 0.9809 |

> ⚠️ **Early Stopping triggered at epoch 15** (no improvement for 4 epochs). Best weights from **Epoch 11** restored.  
> **Final: Acc = 0.9836 · F1 = 0.9842**

---

### Experiment 3 — AE + CNN + BiGRU
*Best Params: LR=0.008950, Weight Decay=0.000001*

| Epoch | Train Loss | Test Loss | Test Acc | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4759 | 0.2231 | 0.9174 | 0.9331 |
| 2 | 0.1487 | 0.1374 | **0.9537** | **0.9599** |
| 3 | 0.1113 | 0.1308 | 0.9572 | 0.9619 |
| 4 | 0.0959 | 0.1626 | 0.9461 | 0.9540 |
| 5 | 0.0891 | 0.1177 | 0.9587 | 0.9635 |
| 6 | 0.0860 | 0.1371 | 0.9489 | 0.9570 |
| 7 | 0.0849 | 0.1706 | 0.9445 | 0.9534 |
| 8 | 0.0822 | 0.1276 | 0.9581 | 0.9633 |
| 9 | 0.0780 | 0.1424 | 0.9515 | 0.9579 |

> ⚠️ **Early Stopping triggered at epoch 9** (no improvement for 4 epochs). Best weights from **Epoch 2** restored.  
> **Final: Acc = 0.9587 · F1 = 0.9635**

---

### Experiment 4 — AE + CNN + BiGRU + Attention (FINAL)
*Best Params: LR=0.001292, Weight Decay=0.000430*

| Epoch | Train Loss | Test Loss | Test Acc | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6563 | 0.4511 | 0.8803 | 0.9063 |
| 2 | 0.2501 | 0.2652 | 0.9204 | 0.9337 |
| 3 | 0.1683 | 0.1589 | 0.9494 | 0.9567 |
| 4 | 0.1356 | 0.1345 | 0.9548 | 0.9614 |
| 5 | 0.1168 | 0.1480 | 0.9501 | 0.9563 |
| 6 | 0.1042 | 0.1208 | 0.9572 | 0.9630 |
| 7 | 0.0956 | 0.1397 | 0.9477 | 0.9560 |
| 8 | 0.0882 | 0.1065 | 0.9632 | 0.9671 |
| 9 | 0.0831 | 0.1006 | 0.9666 | 0.9698 |
| 10 | 0.0800 | 0.0962 | 0.9681 | 0.9714 |
| 11 | 0.0766 | 0.1009 | 0.9667 | 0.9699 |
| 12 | 0.0745 | 0.0754 | **0.9763** | **0.9781** |
| 13 | 0.0711 | 0.1379 | 0.9489 | 0.9576 |
| 14 | 0.0691 | 0.1042 | 0.9648 | 0.9686 |
| 15 | 0.0672 | 0.0866 | 0.9704 | 0.9733 |
| 16 | 0.0667 | 0.0799 | 0.9739 | 0.9761 |

> ⚠️ **Early Stopping triggered at epoch 16** (no improvement for 4 epochs). Best weights from **Epoch 12** restored.  
> **Final: Acc = 0.9763 · F1 = 0.9781**

---

## 5. Ablation Study — Final Results Summary

| Experiment | Configuration | Best Epoch | Accuracy | Weighted F1-Score |
| :--- | :--- | :---: | ---: | ---: |
| Exp 1 | CNN Only | 20 | 0.9747 | 0.9764 |
| **Exp 2** | **CNN + BiGRU** | **11** | **0.9836** | **0.9842** |
| Exp 3 | AE + CNN + BiGRU | 2 | 0.9587 | 0.9635 |
| Exp 4 | AE + CNN + BiGRU + Attention | 12 | 0.9763 | 0.9781 |

> **Best overall model in this run: CNN + BiGRU** (Exp 2) with `Acc = 0.9836`, `F1 = 0.9842`.  
> Confusion matrix plots saved: `confusion_matrix_1_CNN_Only.png`, `confusion_matrix_2_CNN_and_BiGRU.png`, `confusion_matrix_3_AE_and_CNN_and_BiGRU.png`, `confusion_matrix_4_AE_and_CNN_and_BiGRU_and_Attention_(FINAL).png`
