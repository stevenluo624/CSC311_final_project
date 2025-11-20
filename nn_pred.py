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
    
    weights = {
        "W1": np.array(data["W1"]),
        "b1": np.array(data["b1"]),
        "W2": np.array(data["W2"]),
        "b2": np.array(data["b2"]),
        "W3": np.array(data["W3"]),
        "b3": np.array(data["b3"])
    }
    
    config = {
        "vocab_list": data["vocab_list"],
        "binary_cols": data["binary_cols"],
        "best_task_cols": data["best_task_cols"],
        "suboptimal_task_cols": data["suboptimal_task_cols"]
    }
    return weights, config

def _clean_text(text):
    if pd.isna(text): return ""
    s = str(text).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def predict_all(csv_file):
    # Load Configuration
    weights, config = _load_artifacts(ARTIFACTS_FILE)
    vocab_map = {w: i for i, w in enumerate(config["vocab_list"])}
    
    # Load Data
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Data file {csv_file} not found.")
    df = pd.read_csv(csv_file)

    # Numeric Features
    num_cols = ["academic_use_likelihood", "suboptimal_frequency", 
                "reference_expectation", "verify_frequency"]
    for c in num_cols:
        if c not in df.columns: df[c] = 3.0
    X_base_num = (df[num_cols].values - 3.0) / 1.2

    # Count Features
    def get_sum(cols):
        available = [c for c in cols if c in df.columns]
        if not available: return np.zeros(len(df))
        return df[available].sum(axis=1).values
        
    best_counts = get_sum(config["best_task_cols"])
    subopt_counts = get_sum(config["suboptimal_task_cols"])
    X_counts = np.column_stack([best_counts, subopt_counts])
    X_counts = (X_counts - 2.0) / 1.5
    
    X_num = np.hstack([X_base_num, X_counts])

    # Binary Features
    X_bin_list = []
    for col in config["binary_cols"]:
        if col in df.columns:
            X_bin_list.append(df[col].values)
        else:
            X_bin_list.append(np.zeros(len(df)))
    X_bin = np.column_stack(X_bin_list)

    # Bag of Words
    t1 = df['tasks_use_model'].fillna("") if 'tasks_use_model' in df.columns else pd.Series([""] * len(df))
    t2 = df['verify_method'].fillna("") if 'verify_method' in df.columns else pd.Series([""] * len(df))
    full_text = (t1 + " " + t2).apply(_clean_text)

    X_bow = np.zeros((len(df), len(config["vocab_list"])), dtype=np.float32)
    for i, text in enumerate(full_text):
        for word in text.split():
            if word in vocab_map:
                X_bow[i, vocab_map[word]] += 1
    X_bow = np.log1p(X_bow)

    # Combine
    X = np.hstack([X_num, X_bin, X_bow])

    # Inference with Forward Pass
    z1 = np.dot(X, weights["W1"]) + weights["b1"]
    a1 = np.maximum(0, z1)
    
    z2 = np.dot(a1, weights["W2"]) + weights["b2"]
    a2 = np.maximum(0, z2)
    
    logits = np.dot(a2, weights["W3"]) + weights["b3"]
    
    # Get Integer Predictions (0, 1, 2)
    preds_idx = np.argmax(logits, axis=1) 
    
    # Map to Strings
    preds_str = [LABEL_MAP[idx] for idx in preds_idx]

    return preds_str

if __name__ == "__main__":
    pass