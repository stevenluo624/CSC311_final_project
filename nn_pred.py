import numpy as np
import pandas as pd
import json
import re
import os

# Global configuration
ARTIFACTS_FILE = 'model_artifacts.json'
# Final Class Mapping
LABEL_MAP = {
    0: "ChatGPT",
    1: "Claude",
    2: "Gemini"
}

def _load_artifacts(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifacts file {filepath} not found.")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle dynamic layers (lists of arrays)
    weights = [np.array(w) for w in data["weights"]]
    biases = [np.array(b) for b in data["biases"]]
    
    config = {
        "vocab_map": data["vocab_map"],
        "binary_cols": data["binary_cols"],
        # These are needed to calculate counts from raw CSV data
        "best_task_cols": data.get("best_task_cols", []), 
        "suboptimal_task_cols": data.get("suboptimal_task_cols", []),
        "scalers": data["scalers"]
    }
    return weights, biases, config

def _clean_text(text):
    if pd.isna(text): return ""
    s = str(text).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def predict_all(csv_file):
    # Load Configuration from the json file
    weights, biases, config = _load_artifacts(ARTIFACTS_FILE)
    
    # Load Data
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Data file {csv_file} not found.")
    df = pd.read_csv(csv_file)

    # Numeric Features (Scaled using saved statistics)
    num_cols = ["academic_use_likelihood", "suboptimal_frequency", 
                "reference_expectation", "verify_frequency"]
    
    # Ensure columns exist, fill with defaults if missing
    for c in num_cols:
        if c not in df.columns: df[c] = 0.0
            
    X_raw_num = df[num_cols].fillna(0).values
    
    # Apply Robust Scaling: (X - mean) / scale
    mean_num = np.array(config["scalers"]["num_mean"])
    scale_num = np.array(config["scalers"]["num_scale"])
    X_num_scaled = (X_raw_num - mean_num) / scale_num

    # 2. Count Features
    def get_sum(cols):
        # Calculate row-wise sums for specific task columns
        available = [c for c in cols if c in df.columns]
        if not available: return np.zeros(len(df))
        return df[available].sum(axis=1).values
        
    best_counts = get_sum(config["best_task_cols"])
    subopt_counts = get_sum(config["suboptimal_task_cols"])
    X_counts_raw = np.column_stack([best_counts, subopt_counts])
    
    # Apply Robust Scaling to counts
    mean_count = np.array(config["scalers"]["count_mean"])
    scale_count = np.array(config["scalers"]["count_scale"])
    X_counts_scaled = (X_counts_raw - mean_count) / scale_count
    
    # Combine Numeric
    X_numeric_all = np.hstack([X_num_scaled, X_counts_scaled])

    # Binary Features
    X_bin_list = []
    for col in config["binary_cols"]:
        if col in df.columns:
            X_bin_list.append(df[col].fillna(0).values)
        else:
            X_bin_list.append(np.zeros(len(df)))
    X_bin = np.column_stack(X_bin_list)

    # Bag of Words
    t1 = df['tasks_use_model'].fillna("") if 'tasks_use_model' in df.columns else pd.Series([""] * len(df))
    t2 = df['verify_method'].fillna("") if 'verify_method' in df.columns else pd.Series([""] * len(df))
    full_text = (t1 + " " + t2).apply(_clean_text)

    vocab_map = config["vocab_map"]
    X_bow = np.zeros((len(df), len(vocab_map)), dtype=np.float32)
    
    for i, text in enumerate(full_text):
        for word in text.split():
            if word in vocab_map:
                X_bow[i, vocab_map[word]] += 1
    X_bow = np.log1p(X_bow)

    # Combine All Features
    X = np.hstack([X_numeric_all, X_bin, X_bow])

    # 5. Forward Pass
    # Loops through layers saved in the artifacts JSON
    activation = X
    for i, (w, b) in enumerate(zip(weights, biases)):
        z = np.dot(activation, w) + b
        
        # Apply ReLU for all layers except the last one (Logits)
        if i < len(weights) - 1:
            activation = np.maximum(0, z)
        else:
            logits = z
    
    # Get encoded class predictions
    preds_idx = np.argmax(logits, axis=1) 
    
    # Decode to corresponding class
    preds_str = [LABEL_MAP[idx] for idx in preds_idx]

    return preds_str

if __name__ == "__main__":
    try:
        curr_dir = os.getcwd()
        data_dir = 'cleaned_data'
        test_file = 'test_clean.csv'
        path_to_test = os.path.join(curr_dir, data_dir, test_file)
        predictions = predict_all(path_to_test)
        print(predictions[:10])
    except Exception as e:
        print(f"Error: {e}")