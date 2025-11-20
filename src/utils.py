"""
Utility functions for CSE 546 Final Project
"""

import numpy as np
import pandas as pd
import joblib
from datetime import datetime


# Global random state for reproducibility
RANDOM_STATE = 42


def save_results(results, filename, results_dir='results'):
    """
    Save results to pickle file
    
    Parameters:
    -----------
    results : dict or object
        Results to save
    filename : str
        Filename (should end with .pkl)
    results_dir : str
        Results directory path
    """
    import os
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    joblib.dump(results, filepath)
    print(f"Results saved to: {filepath}")


def load_results(filename, results_dir='results'):
    """
    Load results from pickle file
    
    Parameters:
    -----------
    filename : str
        Filename to load
    results_dir : str
        Results directory path
        
    Returns:
    --------
    results : dict or object
        Loaded results
    """
    import os
    filepath = os.path.join(results_dir, filename)
    results = joblib.load(filepath)
    print(f"Results loaded from: {filepath}")
    return results


def print_class_distribution(y, class_names):
    """
    Print detailed class distribution analysis
    
    Parameters:
    -----------
    y : array-like
        Labels
    class_names : array-like
        Class name mapping
    """
    class_counts = pd.Series(y).value_counts().sort_index()
    
    print("Class Distribution Analysis:")
    print("=" * 60)
    for idx, name in enumerate(class_names):
        count = class_counts[idx]
        percentage = (count / len(y)) * 100
        print(f"Class {idx} ({name:10s}): {count:4d} samples ({percentage:5.1f}%)")
    
    max_class = class_counts.max()
    min_class = class_counts.min()
    print("=" * 60)
    print(f"Imbalance Ratio: {max_class/min_class:.2f} (max/min)")
    print(f"Most represented: {class_names[class_counts.idxmax()]} ({max_class} samples)")
    print(f"Least represented: {class_names[class_counts.idxmin()]} ({min_class} samples)")


def log_experiment(exp_num, description, config, results, log_file='experiment_tracker.md'):
    """
    Log experiment to tracker file
    
    Parameters:
    -----------
    exp_num : int
        Experiment number
    description : str
        Experiment description
    config : dict
        Configuration parameters
    results : dict
        Results dictionary
    log_file : str
        Path to experiment log file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = f"""
## Experiment {exp_num:03d}
- **Date**: {timestamp}
- **Description**: {description}
- **Configuration**: {config}
- **Results**:
  - Accuracy: {results.get('val_acc', 'N/A'):.4f}
  - ROC-AUC: {results.get('roc_auc', 'N/A'):.4f}
  - F1-Score: {results.get('f1_macro', 'N/A'):.4f}
- **Status**: Completed

---
"""
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    
    print(f"Experiment {exp_num:03d} logged to {log_file}")

