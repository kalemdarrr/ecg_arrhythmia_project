import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import copy
import os
from models import ECGModel, DenoisingAutoencoder
from data_loader import prepare_data

# Device configuration (MPS - Mac M1, CUDA - Nvidia/Colab, CPU - Default)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device being used: {device}")

def plot_confusion_matrix(targets, preds, exp_name):
    # Generates and saves a confusion matrix plot for the given predictions.
    cm = confusion_matrix(targets, preds)
    class_names = ['N', 'S', 'V', 'F', 'Q']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'Confusion Matrix - {exp_name}')
    
    # Save the plot
    filename = f"confusion_matrix_{exp_name.replace(' ', '_').replace('+', 'and').replace('.', '')}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to {filename}")

def train_autoencoder(model, train_loader, epochs=5, lr=0.001):
    # To train only the Autoencoder. 
    # It can be trained on itself (x -> x) and noisy data (x_noisy -> x).
    print("--- Autoencoder Pre-training Started ---")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, _ in train_loader: # no need for y
            X_batch = X_batch.to(device)
            
            # Add noise (Denoising)
            noise = torch.randn_like(X_batch) * 0.1
            X_noisy = X_batch + noise
            
            optimizer.zero_grad()
            outputs = model(X_noisy)
            loss = criterion(outputs, X_batch) # Compare output with clean signal
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"AE Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
    
    print("--- Autoencoder Training Completed ---")
    return model

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, weight_decay=1e-5, patience=3):
    # Trains the classifier model with Early Stopping mechanism.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    model.to(device)
    
    # --- Early Stopping Variables ---
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0
    best_targets = []
    best_preds = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation / Test
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | "
              f"Test Acc: {acc:.4f} | F1: {f1:.4f}")
              
        # --- Early Stopping Logic ---
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict()) # Save the best model weights
            best_acc = acc
            best_f1 = f1
            best_targets = all_targets
            best_preds = all_preds
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[!] Early stopping triggered! The model did not improve for {patience} epochs.")
                print("Restoring the best weights...\n")
                break
                
    # Restore best model weights when loop ends or early stopping triggers
    model.load_state_dict(best_model_wts)
    return best_acc, best_f1, best_targets, best_preds


def run_ablation_studies():
    print("\nLoading and Preparing Data...")
    
    # SECTION CHANGED FOR FULL TRAINING: (apply_smote=True, max_records=None)
    X_train, X_test, y_train, y_test = prepare_data(apply_smote=True, max_records=None) 
    
    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    # Set batch_size to 1024 for faster GPU processing
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)
    
    # 4 Phase Ablation Study Configurations
    experiments = {
        "1. CNN Only": {"use_ae": False, "use_gru": False, "use_attention": False},
        "2. CNN + BiGRU": {"use_ae": False, "use_gru": True, "use_attention": False},
        "3. AE + CNN + BiGRU": {"use_ae": True, "use_gru": True, "use_attention": False},
        "4. AE + CNN + BiGRU + Attention (FINAL)": {"use_ae": True, "use_gru": True, "use_attention": True}
    }
    
    results = {}
    tuning_results_per_exp = {}
    
    # Pre-train Autoencoder once globally so it can be reused across Optuna trials
    print("\nPre-training Autoencoder globally for models that require it...")
    pretrained_ae = DenoisingAutoencoder()
    pretrained_ae = train_autoencoder(pretrained_ae, train_loader, epochs=10) 
    
    for exp_name, flags in experiments.items():
        print(f"\n{'='*50}\nEXPERIMENT STARTED: {exp_name}\n{'='*50}")
        
        print(f"\n--- Starting Optuna Tuning for {exp_name} ---")
        def objective(trial):
            # Bayesian Optimization search space
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            
            model = ECGModel(num_classes=5, **flags)
            if flags["use_ae"]:
                model.ae.load_state_dict(pretrained_ae.state_dict())
                
            # Patience can be set to 2 during the Optuna phase (for fast tuning)
            acc, f1, _, _ = train_model(model, train_loader, test_loader, epochs=3, lr=lr, weight_decay=wd, patience=2)
            return acc
            
        optuna.logging.set_verbosity(optuna.logging.WARNING) # Keep logs clean
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5) # n_trials=5 as discussed
        
        best_lr = study.best_params["lr"]
        best_wd = study.best_params["weight_decay"]
        print(f"Optuna Best Params for {exp_name} -> LR: {best_lr:.6f}, Weight Decay: {best_wd:.6f} (Val Acc: {study.best_value:.4f})")
        
        tuning_results_per_exp[exp_name] = {
            "best_lr": best_lr,
            "best_wd": best_wd,
            "best_acc": study.best_value
        }
        
        print(f"\n--- Starting Main Training for {exp_name} with Best Params ---")
        # Final Model creation for this experiment
        model = ECGModel(num_classes=5, **flags)
        if flags["use_ae"]:
            model.ae.load_state_dict(pretrained_ae.state_dict())
            
        # Model Training (20 epochs) with best params and Early Stopping (Patience=4)
        acc, f1, targets, preds = train_model(model, train_loader, test_loader, epochs=20, lr=best_lr, weight_decay=best_wd, patience=4) 
        
        # Generate Confusion Matrix for this experiment
        plot_confusion_matrix(targets, preds, exp_name)

        results[exp_name] = {"Accuracy": acc, "F1-Score": f1}
        
    print("\n\n" + "="*50)
    print("ABLATION STUDY RESULTS")
    print("="*50)
    for exp_name, metrics in results.items():
        print(f"{exp_name:40} | Accuracy: {metrics['Accuracy']:.4f} | F1-Score: {metrics['F1-Score']:.4f}")
    print("="*50)

    # Automated Markdown Report Generation
    with open("training_results.md", "w", encoding="utf-8") as f:
        f.write("# Training and Ablation Study Results\n\n")
        
        f.write("## 1. Optuna Hyperparameter Tuning Phase (Independent per Model)\n\n")
        f.write("| Experiment | Best LR | Best Weight Decay | Best Val Accuracy (3 Epochs) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for exp_name, data in tuning_results_per_exp.items():
            f.write(f"| **{exp_name.split('.')[0]}** | `{data['best_lr']:.6f}` | `{data['best_wd']:.6f}` | `{data['best_acc']:.4f}` |\n")
        
        f.write("\n## 2. Ablation Study Results (Main Training - 20 Epochs)\n\n")
        f.write("| Experiment | Configuration | Accuracy | F1-Score |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for exp_name, metrics in results.items():
            f.write(f"| **{exp_name.split('.')[0]}** | {exp_name.split('.')[1].strip()} | `{metrics['Accuracy']:.4f}` | `{metrics['F1-Score']:.4f}` |\n")
            
    print("\n[SUCCESS] Results have been automatically saved to training_results.md")
    print("[SUCCESS] Confusion matrix plots have been generated and saved.")

if __name__ == "__main__":
    run_ablation_studies()
