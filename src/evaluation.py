"""
Evaluation metrics and plotting functions for CSE 546 Final Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report
)


def evaluate_model(model, X, y, cv, model_name="Model"):
    """
    Standard evaluation function for all models
    
    Parameters:
    -----------
    model : sklearn estimator
        Model to evaluate
    X : array-like
        Feature matrix
    y : array-like
        Labels
    cv : cross-validation generator
    model_name : str
        Name for display
        
    Returns:
    --------
    results : dict
        Dictionary with evaluation metrics
    scores : dict
        Raw cross-validation scores
    """
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=['accuracy', 'roc_auc_ovr', 'f1_macro'],
        return_train_score=True,
        n_jobs=-1
    )
    
    results = {
        'model': model_name,
        'train_acc': scores['train_accuracy'].mean(),
        'val_acc': scores['test_accuracy'].mean(),
        'train_acc_std': scores['train_accuracy'].std(),
        'val_acc_std': scores['test_accuracy'].std(),
        'roc_auc': scores['test_roc_auc_ovr'].mean(),
        'f1_macro': scores['test_f1_macro'].mean(),
        'overfit_gap': scores['train_accuracy'].mean() - scores['test_accuracy'].mean()
    }
    
    return results, scores


def plot_learning_curve(estimator, X, y, cv, title="Learning Curve", figsize=(10, 6)):
    """
    Generate learning curve plots
    
    Parameters:
    -----------
    estimator : sklearn estimator
    X : array-like
        Feature matrix
    y : array-like
        Labels
    cv : cross-validation generator
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib figure
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='orange')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    ax.plot(train_sizes, val_mean, 's-', color='orange', label='Validation score')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    return fig


def save_figure(fig, name, report_num=1, output_dir='results/figures'):
    """
    Save figure with consistent naming
    
    Parameters:
    -----------
    fig : matplotlib figure
    name : str
        Base name for the figure
    report_num : int
        Report number (1 or 2)
    output_dir : str
        Output directory path
    """
    import os
    report_dir = f'report{report_num}' if report_num else 'final'
    full_dir = os.path.join(output_dir, report_dir)
    os.makedirs(full_dir, exist_ok=True)
    
    filename = f'{full_dir}/{name}.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {filename}")

