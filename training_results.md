# Training and Ablation Study Results

## 1. Data Preparation & Distribution

- **Hardware Acceleration**: `CUDA`
- **Total Extracted Heartbeats**: 109,452
- **Initial Class Distribution (Before SMOTE)**: `[90594, 2781, 7235, 802, 8040]`
- **SMOTE Applied**: To resolve severe class imbalance, SMOTE was applied to the training set.
- **Class Distribution (After SMOTE - Training)**: `[72475, 72475, 72475, 72475, 72475]`

### Dataset Splits
- **Training Set (`X_train`)**: `(362375, 1, 250)`
- **Training Labels (`y_train`)**: `(362375,)`
- **Test Set (`X_test`)**: `(21891, 1, 250)`
- **Test Labels (`y_test`)**: `(21891,)`

---

## 2. Ablation Study Summary

The ablation study sequentially evaluates the contribution of each architectural component. The models were evaluated on the imbalanced test set (20% of original data) after being trained on the SMOTE-augmented training set.

| Experiment | Configuration | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **Experiment 1** | CNN Only | `0.9853` | `0.9858` |
| **Experiment 2** | CNN + BiGRU | `0.9844` | `0.9851` |
| **Experiment 3** | AE + CNN + BiGRU | `0.9839` | `0.9848` |
| **Experiment 4** | **AE + CNN + BiGRU + Attention (FINAL)** | `0.9852` | `0.9858` |

> **Note**: While the simple 1D CNN baseline achieved very high performance (`0.9853`), the final architecture incorporating Denoising Autoencoder (AE), BiGRU, and Attention mechanism maintained a nearly identical performance (`0.9852`) while providing much stronger theoretical robustness against noise and capturing temporal context effectively.

---

## 3. Detailed Training Logs

<details>
<summary><b>Click to expand detailed epoch-by-epoch logs</b></summary>

### Experiment 1: CNN Only
```text
Epoch [1/20] | Train Loss: 0.3024 | Test Loss: 0.1969 | Test Acc: 0.9427 | F1: 0.9500
Epoch [2/20] | Train Loss: 0.1715 | Test Loss: 0.1783 | Test Acc: 0.9435 | F1: 0.9512
Epoch [3/20] | Train Loss: 0.1312 | Test Loss: 0.1253 | Test Acc: 0.9593 | F1: 0.9643
Epoch [4/20] | Train Loss: 0.1038 | Test Loss: 0.0997 | Test Acc: 0.9690 | F1: 0.9720
Epoch [5/20] | Train Loss: 0.0832 | Test Loss: 0.0811 | Test Acc: 0.9759 | F1: 0.9774
Epoch [6/20] | Train Loss: 0.0717 | Test Loss: 0.0716 | Test Acc: 0.9802 | F1: 0.9812
Epoch [7/20] | Train Loss: 0.0624 | Test Loss: 0.0706 | Test Acc: 0.9799 | F1: 0.9810
Epoch [8/20] | Train Loss: 0.0549 | Test Loss: 0.0755 | Test Acc: 0.9777 | F1: 0.9791
Epoch [9/20] | Train Loss: 0.0486 | Test Loss: 0.0557 | Test Acc: 0.9860 | F1: 0.9862
Epoch [10/20] | Train Loss: 0.0439 | Test Loss: 0.0741 | Test Acc: 0.9781 | F1: 0.9796
Epoch [11/20] | Train Loss: 0.0400 | Test Loss: 0.0658 | Test Acc: 0.9815 | F1: 0.9823
Epoch [12/20] | Train Loss: 0.0372 | Test Loss: 0.0674 | Test Acc: 0.9816 | F1: 0.9826
Epoch [13/20] | Train Loss: 0.0347 | Test Loss: 0.0675 | Test Acc: 0.9815 | F1: 0.9825
Epoch [14/20] | Train Loss: 0.0333 | Test Loss: 0.0621 | Test Acc: 0.9843 | F1: 0.9848
Epoch [15/20] | Train Loss: 0.0307 | Test Loss: 0.0571 | Test Acc: 0.9861 | F1: 0.9864
Epoch [16/20] | Train Loss: 0.0299 | Test Loss: 0.0572 | Test Acc: 0.9862 | F1: 0.9865
Epoch [17/20] | Train Loss: 0.0288 | Test Loss: 0.0638 | Test Acc: 0.9841 | F1: 0.9846
Epoch [18/20] | Train Loss: 0.0283 | Test Loss: 0.0594 | Test Acc: 0.9857 | F1: 0.9861
Epoch [19/20] | Train Loss: 0.0268 | Test Loss: 0.0622 | Test Acc: 0.9856 | F1: 0.9859
Epoch [20/20] | Train Loss: 0.0265 | Test Loss: 0.0590 | Test Acc: 0.9853 | F1: 0.9858
```

### Experiment 2: CNN + BiGRU
```text
Epoch [1/20] | Train Loss: 0.2093 | Test Loss: 0.1349 | Test Acc: 0.9554 | F1: 0.9609
Epoch [2/20] | Train Loss: 0.0885 | Test Loss: 0.0855 | Test Acc: 0.9739 | F1: 0.9761
Epoch [3/20] | Train Loss: 0.0650 | Test Loss: 0.0651 | Test Acc: 0.9807 | F1: 0.9817
Epoch [4/20] | Train Loss: 0.0539 | Test Loss: 0.0988 | Test Acc: 0.9686 | F1: 0.9719
Epoch [5/20] | Train Loss: 0.0458 | Test Loss: 0.0597 | Test Acc: 0.9840 | F1: 0.9847
Epoch [6/20] | Train Loss: 0.0413 | Test Loss: 0.0747 | Test Acc: 0.9797 | F1: 0.9810
Epoch [7/20] | Train Loss: 0.0375 | Test Loss: 0.0576 | Test Acc: 0.9855 | F1: 0.9859
Epoch [8/20] | Train Loss: 0.0352 | Test Loss: 0.0615 | Test Acc: 0.9837 | F1: 0.9843
Epoch [9/20] | Train Loss: 0.0329 | Test Loss: 0.0772 | Test Acc: 0.9808 | F1: 0.9818
Epoch [10/20] | Train Loss: 0.0312 | Test Loss: 0.0649 | Test Acc: 0.9847 | F1: 0.9855
Epoch [11/20] | Train Loss: 0.0297 | Test Loss: 0.0664 | Test Acc: 0.9837 | F1: 0.9844
Epoch [12/20] | Train Loss: 0.0280 | Test Loss: 0.0538 | Test Acc: 0.9877 | F1: 0.9879
Epoch [13/20] | Train Loss: 0.0276 | Test Loss: 0.0645 | Test Acc: 0.9845 | F1: 0.9852
Epoch [14/20] | Train Loss: 0.0270 | Test Loss: 0.0602 | Test Acc: 0.9852 | F1: 0.9857
Epoch [15/20] | Train Loss: 0.0260 | Test Loss: 0.0842 | Test Acc: 0.9803 | F1: 0.9817
Epoch [16/20] | Train Loss: 0.0258 | Test Loss: 0.0743 | Test Acc: 0.9813 | F1: 0.9825
Epoch [17/20] | Train Loss: 0.0250 | Test Loss: 0.0761 | Test Acc: 0.9805 | F1: 0.9818
Epoch [18/20] | Train Loss: 0.0233 | Test Loss: 0.0674 | Test Acc: 0.9867 | F1: 0.9871
Epoch [19/20] | Train Loss: 0.0238 | Test Loss: 0.0599 | Test Acc: 0.9852 | F1: 0.9860
Epoch [20/20] | Train Loss: 0.0232 | Test Loss: 0.0639 | Test Acc: 0.9844 | F1: 0.9851
```

### Experiment 3: AE + CNN + BiGRU
**Autoencoder Pre-training Phase:**
```text
AE Epoch [1/10], Loss: 0.0023
AE Epoch [2/10], Loss: 0.0014
AE Epoch [3/10], Loss: 0.0013
...
AE Epoch [10/10], Loss: 0.0013
```

**Classifier Training:**
```text
Epoch [1/20] | Train Loss: 0.2076 | Test Loss: 0.0979 | Test Acc: 0.9670 | F1: 0.9703
Epoch [2/20] | Train Loss: 0.0842 | Test Loss: 0.0810 | Test Acc: 0.9748 | F1: 0.9770
Epoch [3/20] | Train Loss: 0.0628 | Test Loss: 0.0868 | Test Acc: 0.9751 | F1: 0.9770
Epoch [4/20] | Train Loss: 0.0533 | Test Loss: 0.0808 | Test Acc: 0.9727 | F1: 0.9752
Epoch [5/20] | Train Loss: 0.0467 | Test Loss: 0.0760 | Test Acc: 0.9746 | F1: 0.9767
Epoch [6/20] | Train Loss: 0.0429 | Test Loss: 0.0637 | Test Acc: 0.9825 | F1: 0.9835
Epoch [7/20] | Train Loss: 0.0389 | Test Loss: 0.0709 | Test Acc: 0.9796 | F1: 0.9810
Epoch [8/20] | Train Loss: 0.0357 | Test Loss: 0.0687 | Test Acc: 0.9799 | F1: 0.9814
Epoch [9/20] | Train Loss: 0.0340 | Test Loss: 0.0572 | Test Acc: 0.9839 | F1: 0.9847
Epoch [10/20] | Train Loss: 0.0330 | Test Loss: 0.0639 | Test Acc: 0.9818 | F1: 0.9827
Epoch [11/20] | Train Loss: 0.0311 | Test Loss: 0.0693 | Test Acc: 0.9802 | F1: 0.9817
Epoch [12/20] | Train Loss: 0.0304 | Test Loss: 0.0543 | Test Acc: 0.9854 | F1: 0.9862
Epoch [13/20] | Train Loss: 0.0292 | Test Loss: 0.0604 | Test Acc: 0.9841 | F1: 0.9849
Epoch [14/20] | Train Loss: 0.0285 | Test Loss: 0.0631 | Test Acc: 0.9839 | F1: 0.9848
Epoch [15/20] | Train Loss: 0.0274 | Test Loss: 0.0605 | Test Acc: 0.9849 | F1: 0.9855
Epoch [16/20] | Train Loss: 0.0267 | Test Loss: 0.0546 | Test Acc: 0.9871 | F1: 0.9874
Epoch [17/20] | Train Loss: 0.0266 | Test Loss: 0.0647 | Test Acc: 0.9813 | F1: 0.9827
Epoch [18/20] | Train Loss: 0.0256 | Test Loss: 0.0623 | Test Acc: 0.9815 | F1: 0.9832
Epoch [19/20] | Train Loss: 0.0249 | Test Loss: 0.0708 | Test Acc: 0.9815 | F1: 0.9826
Epoch [20/20] | Train Loss: 0.0248 | Test Loss: 0.0603 | Test Acc: 0.9839 | F1: 0.9848
```

### Experiment 4: AE + CNN + BiGRU + Attention (FINAL)
```text
Epoch [1/20] | Train Loss: 0.3069 | Test Loss: 0.1388 | Test Acc: 0.9559 | F1: 0.9609
Epoch [2/20] | Train Loss: 0.1058 | Test Loss: 0.1129 | Test Acc: 0.9571 | F1: 0.9624
Epoch [3/20] | Train Loss: 0.0768 | Test Loss: 0.1271 | Test Acc: 0.9498 | F1: 0.9584
Epoch [4/20] | Train Loss: 0.0623 | Test Loss: 0.1259 | Test Acc: 0.9492 | F1: 0.9576
Epoch [5/20] | Train Loss: 0.0537 | Test Loss: 0.0811 | Test Acc: 0.9712 | F1: 0.9739
Epoch [6/20] | Train Loss: 0.0480 | Test Loss: 0.0715 | Test Acc: 0.9764 | F1: 0.9781
Epoch [7/20] | Train Loss: 0.0443 | Test Loss: 0.0732 | Test Acc: 0.9758 | F1: 0.9777
Epoch [8/20] | Train Loss: 0.0399 | Test Loss: 0.0636 | Test Acc: 0.9799 | F1: 0.9810
Epoch [9/20] | Train Loss: 0.0372 | Test Loss: 0.0581 | Test Acc: 0.9815 | F1: 0.9825
Epoch [10/20] | Train Loss: 0.0361 | Test Loss: 0.0815 | Test Acc: 0.9728 | F1: 0.9754
Epoch [11/20] | Train Loss: 0.0341 | Test Loss: 0.0922 | Test Acc: 0.9679 | F1: 0.9718
Epoch [12/20] | Train Loss: 0.0329 | Test Loss: 0.0759 | Test Acc: 0.9765 | F1: 0.9782
Epoch [13/20] | Train Loss: 0.0309 | Test Loss: 0.0625 | Test Acc: 0.9793 | F1: 0.9808
Epoch [14/20] | Train Loss: 0.0298 | Test Loss: 0.0521 | Test Acc: 0.9856 | F1: 0.9861
Epoch [15/20] | Train Loss: 0.0297 | Test Loss: 0.0604 | Test Acc: 0.9816 | F1: 0.9827
Epoch [16/20] | Train Loss: 0.0282 | Test Loss: 0.0672 | Test Acc: 0.9788 | F1: 0.9803
Epoch [17/20] | Train Loss: 0.0274 | Test Loss: 0.0634 | Test Acc: 0.9813 | F1: 0.9826
Epoch [18/20] | Train Loss: 0.0274 | Test Loss: 0.0587 | Test Acc: 0.9822 | F1: 0.9831
Epoch [19/20] | Train Loss: 0.0262 | Test Loss: 0.0613 | Test Acc: 0.9820 | F1: 0.9832
Epoch [20/20] | Train Loss: 0.0259 | Test Loss: 0.0508 | Test Acc: 0.9852 | F1: 0.9858
```
</details>
