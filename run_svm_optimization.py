"""
SVM Optimization for Report 1
Experiment 005: Systematic SVM parameter tuning
Run this script after preprocessing experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

from src.preprocessing import load_data
from src.evaluation import save_figure
from src.utils import RANDOM_STATE, save_results, log_experiment

# Setup
np.random.seed(RANDOM_STATE)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("CSE 546 SVM OPTIMIZATION (Experiment 005)")
print("="*70)

# Load data
X_train, y_train, _, _, class_names = load_data('data/')
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

print(f"\nData loaded: {X_train.shape}")
print(f"Target: Beat 87.16% baseline\n")

#############################################################################
# EXPERIMENT 005: SVM OPTIMIZATION
#############################################################################

print("="*70)
print("EXPERIMENT 005: SVM PARAMETER OPTIMIZATION")
print("="*70)

# Pipeline with StandardScaler (best from Exp 002)
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=RANDOM_STATE, probability=True))
])

# Parameter grid
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['rbf', 'poly', 'sigmoid'],
    'svm__gamma': ['scale', 'auto']
}

print("\nRunning GridSearchCV...")
print("This may take 5-10 minutes...")

# GridSearchCV
grid_search = GridSearchCV(
    svm_pipeline, param_grid, cv=cv,
    scoring=['accuracy', 'roc_auc_ovr', 'f1_macro'],
    refit='accuracy',
    return_train_score=True,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Extract results
results_df = pd.DataFrame(grid_search.cv_results_)

print(f"\n{'='*70}")
print(f"BEST CONFIGURATION:")
print(f"{'='*70}")
print(f"Parameters: {grid_search.best_params_}")
print(f"CV Accuracy: {grid_search.best_score_:.4f}")
print(f"Improvement: {(grid_search.best_score_ - 0.8716)*100:+.2f}%")

# Analyze best configuration
best_idx = grid_search.best_index_
best_row = results_df.iloc[best_idx]

print(f"\nDetailed Metrics:")
print(f"  Train Accuracy: {best_row['mean_train_accuracy']:.4f}")
print(f"  Val Accuracy:   {best_row['mean_test_accuracy']:.4f}")
print(f"  ROC-AUC:        {best_row['mean_test_roc_auc_ovr']:.4f}")
print(f"  F1-Score:       {best_row['mean_test_f1_macro']:.4f}")
print(f"  Overfit Gap:    {(best_row['mean_train_accuracy'] - best_row['mean_test_accuracy']):.4f}")

# Fold consistency
fold_scores = []
for i in range(4):
    fold_scores.append(best_row[f'split{i}_test_accuracy'])
print(f"\nFold Scores: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Std Dev:     {np.std(fold_scores):.4f}")

#############################################################################
# FIGURE 7: LEARNING CURVES
#############################################################################

print("\nGenerating learning curves...")

best_estimator = grid_search.best_estimator_

train_sizes, train_scores, val_scores = learning_curve(
    best_estimator, X_train, y_train, cv=cv, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                 alpha=0.1, color='blue')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                 alpha=0.1, color='orange')
ax.plot(train_sizes, train_mean, 'o-', color='blue', 
        label='Training score', linewidth=2, markersize=6)
ax.plot(train_sizes, val_mean, 's-', color='orange', 
        label='Validation score', linewidth=2, markersize=6)

ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Figure 7: SVM Learning Curves (Best Configuration)', 
             fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, 'figure7_svm_learning_curves', report_num=1)
plt.close()

#############################################################################
# FIGURE 8: PARAMETER IMPACT (C parameter)
#############################################################################

print("Generating parameter impact plot...")

# Focus on RBF kernel results
rbf_results = results_df[results_df['param_svm__kernel'] == 'rbf'].copy()

fig, ax = plt.subplots(figsize=(10, 6))

for gamma in ['scale', 'auto']:
    gamma_data = rbf_results[rbf_results['param_svm__gamma'] == gamma]
    c_values = gamma_data['param_svm__C'].values
    train_acc = gamma_data['mean_train_accuracy'].values
    val_acc = gamma_data['mean_test_accuracy'].values
    
    ax.plot(c_values, train_acc, 'o--', label=f'Train (gamma={gamma})', 
            linewidth=2, markersize=8, alpha=0.7)
    ax.plot(c_values, val_acc, 's-', label=f'Val (gamma={gamma})', 
            linewidth=2, markersize=8)

ax.set_xscale('log')
ax.set_xlabel('C Parameter', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Figure 8: Impact of C Parameter on SVM Performance (RBF Kernel)', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, 'figure8_svm_parameter_impact', report_num=1)
plt.close()

#############################################################################
# FIGURE 9: CONFUSION MATRIX
#############################################################################

print("Generating confusion matrix...")

# Get predictions using cross_val_predict
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(best_estimator, X_train, y_train, cv=cv)

# Confusion matrix
cm = confusion_matrix(y_train, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
ax.set_title('Figure 9: Confusion Matrix - Optimized SVM', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_figure(fig, 'figure9_svm_confusion_matrix', report_num=1)
plt.close()

# Analyze confusion
print("\nConfusion Analysis:")
for i, true_class in enumerate(class_names):
    for j, pred_class in enumerate(class_names):
        if i != j and cm[i, j] > 10:  # Significant confusion
            print(f"  {true_class} confused with {pred_class}: {cm[i, j]} samples")

#############################################################################
# SAVE RESULTS
#############################################################################

# Save best model
joblib.dump(best_estimator, 'models/best_svm.pkl')
print("\nBest model saved to models/best_svm.pkl")

# Prepare detailed results
experiment_005 = {
    'experiment_id': '005',
    'description': 'SVM parameter optimization',
    'best_params': grid_search.best_params_,
    'best_score': grid_search.best_score_,
    'best_estimator': best_estimator,
    'cv_results': results_df,
    'confusion_matrix': cm,
    'metrics': {
        'train_accuracy': best_row['mean_train_accuracy'],
        'val_accuracy': best_row['mean_test_accuracy'],
        'roc_auc': best_row['mean_test_roc_auc_ovr'],
        'f1_score': best_row['mean_test_f1_macro'],
        'overfit_gap': best_row['mean_train_accuracy'] - best_row['mean_test_accuracy'],
        'fold_std': np.std(fold_scores)
    },
    'status': 'completed'
}

save_results(experiment_005, 'svm_optimization_results.pkl', 
             results_dir='results/classifiers')

log_experiment(
    exp_num=5,
    description=f"SVM Optimized: {grid_search.best_params_['svm__kernel']} kernel",
    config=grid_search.best_params_,
    results={'val_acc': best_row['mean_test_accuracy'], 
             'roc_auc': best_row['mean_test_roc_auc_ovr'],
             'f1_macro': best_row['mean_test_f1_macro']},
    log_file='experiment_tracker.md'
)

#############################################################################
# SUMMARY
#############################################################################

print("\n" + "="*70)
print("SVM OPTIMIZATION COMPLETE!")
print("="*70)
print(f"\nBest Configuration:")
print(f"  Kernel: {grid_search.best_params_['svm__kernel']}")
print(f"  C: {grid_search.best_params_['svm__C']}")
print(f"  Gamma: {grid_search.best_params_['svm__gamma']}")

print(f"\nPerformance:")
print(f"  CV Accuracy: {best_row['mean_test_accuracy']:.4f}")
print(f"  ROC-AUC:     {best_row['mean_test_roc_auc_ovr']:.4f}")
print(f"  F1-Score:    {best_row['mean_test_f1_macro']:.4f}")
print(f"  Improvement: {(best_row['mean_test_accuracy'] - 0.8716)*100:+.2f}%")

print(f"\nFigures generated:")
print(f"  - Figure 7: results/figures/report1/figure7_svm_learning_curves.png")
print(f"  - Figure 8: results/figures/report1/figure8_svm_parameter_impact.png")
print(f"  - Figure 9: results/figures/report1/figure9_svm_confusion_matrix.png")

print(f"\nModel saved: models/best_svm.pkl")
print(f"Results saved: results/classifiers/svm_optimization_results.pkl")
print(f"Experiment tracker updated!")
print("="*70)

