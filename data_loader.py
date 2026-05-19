import os
import wfdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# AAMI 5-class mapping (from MIT-BIH annotations to AAMI standards)
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,         # Supraventricular (S)
    'V': 2, 'E': 2,                         # Ventricular (V)
    'F': 3,                                 # Fusion (F)
    '/': 4, 'f': 4, 'Q': 4                  # Unknown (Q)
}

# Symbols that are considered valid and exist in the mapping above
VALID_SYMBOLS = list(AAMI_MAPPING.keys())

def download_mitdb(dl_dir='./mitdb'):
    """Downloads the MIT-BIH dataset."""
    if not os.path.exists(dl_dir):
        os.makedirs(dl_dir)
        print("Downloading MIT-BIH Arrhythmia Database...")
        wfdb.dl_database('mitdb', dl_dir=dl_dir)
        print("Download complete.")
    else:
        print("Dataset already exists.")

def load_data(data_dir='./mitdb', window_size=250, max_records=None):
    """
    Segments the signals centered around R-peaks and assigns labels.
    window_size: Total number of samples (e.g., 90 before, 160 after)
    """
    records = wfdb.get_record_list('mitdb')
    if max_records is not None:
        records = records[:max_records]
    
    X = []
    y = []
    
    # Window to be extracted around the R peak (360Hz frequency -> 250 samples is approx 0.7 seconds)
    before = 90
    after = window_size - before
    
    print("Processing records (Windowing and Labeling)...")
    for record in records:
        record_path = os.path.join(data_dir, record)
        try:
            # Read signal and annotations
            signal, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            # Channel 0 (MLII) is typically used
            lead_0 = signal[:, 0]
            
            peaks = annotation.sample
            symbols = annotation.symbol
            
            for i, peak in enumerate(peaks):
                symbol = symbols[i]
                
                # Check if it is a valid class and window boundaries are valid
                if symbol in VALID_SYMBOLS and (peak - before) >= 0 and (peak + after) < len(lead_0):
                    window = lead_0[peak - before : peak + after]
                    # Simple Min-Max Normalization (per heartbeat)
                    window_normalized = minmax_scale(window)
                    
                    X.append(window_normalized)
                    y.append(AAMI_MAPPING[symbol])
                    
        except Exception as e:
            print(f"Error: {record} could not be processed. ({e})")
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Total extracted heartbeats: {len(X)}")
    print(f"Class distribution (Before SMOTE): {np.bincount(y)}")
    
    return X, y

def prepare_data(data_dir='./mitdb', test_size=0.2, apply_smote=True, max_records=None):
    """
    Downloads, processes, and splits the dataset, optionally applying SMOTE.
    Formats it to (N, C, L) suitable for PyTorch 1D CNN input.
    """
    download_mitdb(data_dir)
    X, y = load_data(data_dir, max_records=max_records)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    
    if apply_smote:
        print("Applying SMOTE to the training data (to resolve class imbalance)...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"Class distribution (After SMOTE - Training): {np.bincount(y_train)}")
    
    # Format for PyTorch 1D CNN input: (Batch, Channels, Length) -> (N, 1, 250)
    X_train = X_train.reshape(-1, 1, X_train.shape[1])
    X_test = X_test.reshape(-1, 1, X_test.shape[1])
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Can be run for testing purposes
    X_tr, X_te, y_tr, y_te = prepare_data(apply_smote=False) # SMOTE disabled for quick test
