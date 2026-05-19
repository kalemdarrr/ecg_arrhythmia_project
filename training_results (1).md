# Training and Ablation Study Results

## 1. Optuna Hyperparameter Tuning Phase (Independent per Model)

| Experiment | Best LR | Best Weight Decay | Best Val Accuracy (3 Epochs) |
| :--- | :--- | :--- | :--- |
| **1** | `0.000787` | `0.000300` | `0.9654` |
| **2** | `0.001576` | `0.000003` | `0.9823` |
| **3** | `0.001476` | `0.000005` | `0.9746` |
| **4** | `0.000733` | `0.000035` | `0.9667` |

## 2. Ablation Study Results (Main Training - 20 Epochs)

| Experiment | Configuration | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **1** | CNN Only | `0.9804` | `0.9812` |
| **2** | CNN + BiGRU | `0.9800` | `0.9812` |
| **3** | AE + CNN + BiGRU | `0.9809` | `0.9821` |
| **4** | AE + CNN + BiGRU + Attention (FINAL) | `0.9795` | `0.9808` |
