# Training and Ablation Study Results

## 1. Optuna Hyperparameter Tuning Phase (Independent per Model)

| Experiment | Best LR | Best Weight Decay | Best Val Accuracy (3 Epochs) |
| :--- | :--- | :--- | :--- |
| **1** | `0.004534` | `0.000372` | `0.9577` |
| **2** | `0.000674` | `0.000053` | `0.9633` |
| **3** | `0.008950` | `0.000001` | `0.9644` |
| **4** | `0.001292` | `0.000430` | `0.9375` |

## 2. Ablation Study Results (Main Training - 20 Epochs)

| Experiment | Configuration | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **1** | CNN Only | `0.9747` | `0.9764` |
| **2** | CNN + BiGRU | `0.9836` | `0.9842` |
| **3** | AE + CNN + BiGRU | `0.9587` | `0.9635` |
| **4** | AE + CNN + BiGRU + Attention (FINAL) | `0.9763` | `0.9781` |
