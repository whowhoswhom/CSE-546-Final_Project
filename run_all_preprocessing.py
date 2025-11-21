"""
Complete Preprocessing Experiments for Report 1
Experiments 002-004: Normalization, PCA, Feature Selection
Run this script to generate all results and figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

# Local imports
from src.preprocessing import load_data
from src.evaluation import save_figure
from src.utils import RANDOM_STATE, save_results, log_experiment

# Setup
np.random.seed(RANDOM_STATE)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("CSE 546 PREPROCESSING EXPERIMENTS (002-004)")
print("="*70)

# Load data
X_train, y_train, _, _, class_names = load_data('data/')
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

print(f"\nData loaded: {X_train.shape}")
print(f"Baseline to beat: 87.16%\n")

#############################################################################
# EXPERIMENT 002: NORMALIZATION COMPARISON
#############################################################################

print("="*70)
print("EXPERIMENT 002: NORMALIZATION COMPARISON")
print("="*70)

scalers = {
    'None': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

classifiers = {
    'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
    'SVM (RBF)': SVC(C=1.0, kernel='rbf', random_state=RANDOM_STATE)
}

normalization_results = []

for scaler_name, scaler in scalers.items():
    for clf_name, clf in classifiers.items():
        print(f"\nTesting {scaler_name:15s} + {clf_name}...", end=" ")
        
        if scaler is None:
            model = clf
        else:
            model = Pipeline([
                ('scaler', scaler),
                ('classifier', clf)
            ])
        
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=['accuracy', 'roc_auc_ovr', 'f1_macro'],
            return_train_score=True,
            n_jobs=-1
        )
        
        result = {
            'Scaler': scaler_name,
            'Classifier': clf_name,
            'Val_Accuracy': scores['test_accuracy'].mean(),
            'Val_Acc_Std': scores['test_accuracy'].std(),
            'Train_Accuracy': scores['train_accuracy'].mean(),
            'ROC_AUC': scores['test_roc_auc_ovr'].mean(),
            'F1_Score': scores['test_f1_macro'].mean(),
            'Overfit_Gap': scores['train_accuracy'].mean() - scores['test_accuracy'].mean()
        }
        
        normalization_results.append(result)
        print(f"Acc: {result['Val_Accuracy']:.4f}")

norm_df = pd.DataFrame(normalization_results)

# Figure 3: Normalization Comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i, clf in enumerate(classifiers.keys()):
    clf_data = norm_df[norm_df['Classifier'] == clf]
    x_pos = np.arange(len(scalers))
    offset = -0.2 if i == 0 else 0.2
    color = 'steelblue' if i == 0 else 'coral'
    
    axes[0].bar(x_pos + offset, clf_data['Val_Accuracy'], width=0.35,
                label=clf, alpha=0.8, color=color, edgecolor='black')

axes[0].axhline(y=0.8716, color='red', linestyle='--', linewidth=2,
                label='Baseline (87.16%)', alpha=0.7)
axes[0].set_xlabel('Normalization Method', fontsize=12, fontweight='bold')
axes[0].set_ylabel('CV Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Impact of Normalization', fontsize=13, fontweight='bold')
axes[0].set_xticks(range(len(scalers)))
axes[0].set_xticklabels(scalers.keys(), rotation=45, ha='right')
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0.80, 0.92])

for i, clf in enumerate(classifiers.keys()):
    clf_data = norm_df[norm_df['Classifier'] == clf]
    marker = 'o' if i == 0 else 's'
    color = 'steelblue' if i == 0 else 'coral'
    axes[1].plot(range(len(scalers)), clf_data['Overfit_Gap'], marker=marker,
                 label=clf, linewidth=2, markersize=8, color=color)

axes[1].set_xlabel('Normalization Method', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Overfitting Gap', fontsize=12, fontweight='bold')
axes[1].set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
axes[1].set_xticks(range(len(scalers)))
axes[1].set_xticklabels(scalers.keys(), rotation=45, ha='right')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.suptitle('Figure 3: Normalization Method Comparison', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
save_figure(fig, 'figure3_normalization_comparison', report_num=1)
plt.close()

best_config = norm_df.loc[norm_df['Val_Accuracy'].idxmax()]
print(f"\n{'='*70}")
print(f"Best: {best_config['Scaler']} + {best_config['Classifier']}")
print(f"Accuracy: {best_config['Val_Accuracy']:.4f} ({(best_config['Val_Accuracy']-0.8716)*100:+.2f}%)")
print(f"{'='*70}")

# Save results
experiment_002 = {
    'experiment_id': '002',
    'description': 'Normalization comparison',
    'results_df': norm_df,
    'best_config': best_config.to_dict(),
    'status': 'completed'
}
save_results(experiment_002, 'normalization_results.pkl', 
             results_dir='results/preprocessing')

log_experiment(
    exp_num=2,
    description=f"Normalization: Best={best_config['Scaler']}",
    config={'scaler': best_config['Scaler'], 'classifier': best_config['Classifier']},
    results={'val_acc': best_config['Val_Accuracy'], 
             'roc_auc': best_config['ROC_AUC'],
             'f1_macro': best_config['F1_Score']},
    log_file='experiment_tracker.md'
)

#############################################################################
# EXPERIMENT 003: PCA ANALYSIS
#############################################################################

print("\n" + "="*70)
print("EXPERIMENT 003: PCA ANALYSIS")
print("="*70)

# Use best scaler from Exp 002
best_scaler_name = best_config['Scaler']
if best_scaler_name == 'None':
    best_scaler = None
else:
    best_scaler = scalers[best_scaler_name]

# PCA variance analysis
if best_scaler:
    X_scaled = best_scaler.fit_transform(X_train)
else:
    X_scaled = X_train.values

pca_full = PCA()
pca_full.fit(X_scaled)

# Calculate components for variance thresholds
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_99 = np.argmax(cumulative_variance >= 0.99) + 1

print(f"\nComponents for 95% variance: {n_95}")
print(f"Components for 99% variance: {n_99}")

# Figure 4: PCA Variance Analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Scree plot
axes[0].plot(range(1, 51), pca_full.explained_variance_ratio_[:50], 
             'bo-', linewidth=2, markersize=6)
axes[0].set_xlabel('Component Number', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
axes[0].set_title('Figure 4a: Scree Plot (First 50 Components)', 
                  fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3)

# Cumulative variance
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
             linewidth=2, color='steelblue')
axes[1].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% variance')
axes[1].axvline(x=n_95, color='red', linestyle=':', alpha=0.5)
axes[1].axhline(y=0.99, color='orange', linestyle='--', linewidth=2, label='99% variance')
axes[1].axvline(x=n_99, color='orange', linestyle=':', alpha=0.5)
axes[1].set_xlabel('Number of Components', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cumulative Variance', fontsize=12, fontweight='bold')
axes[1].set_title('Figure 4b: Cumulative Variance Explained', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.suptitle('Figure 4: PCA Variance Analysis', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
save_figure(fig, 'figure4_pca_variance', report_num=1)
plt.close()

# Test different n_components
n_components_list = [50, 100, 150, n_95, n_99]
pca_results = []

print("\nTesting PCA configurations...")
for n_comp in n_components_list:
    print(f"  n_components={n_comp:3d}...", end=" ")
    
    if best_scaler:
        pipeline = Pipeline([
            ('scaler', best_scaler),
            ('pca', PCA(n_components=n_comp, random_state=RANDOM_STATE)),
            ('classifier', SVC(C=1.0, kernel='rbf', random_state=RANDOM_STATE))
        ])
    else:
        pipeline = Pipeline([
            ('pca', PCA(n_components=n_comp, random_state=RANDOM_STATE)),
            ('classifier', SVC(C=1.0, kernel='rbf', random_state=RANDOM_STATE))
        ])
    
    scores = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring=['accuracy', 'roc_auc_ovr', 'f1_macro'],
        n_jobs=-1
    )
    
    result = {
        'n_components': n_comp,
        'accuracy': scores['test_accuracy'].mean(),
        'std': scores['test_accuracy'].std(),
        'roc_auc': scores['test_roc_auc_ovr'].mean(),
        'f1_score': scores['test_f1_macro'].mean()
    }
    pca_results.append(result)
    print(f"Acc: {result['accuracy']:.4f}")

pca_df = pd.DataFrame(pca_results)

# Figure 5: PCA Performance
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(pca_df['n_components'], pca_df['accuracy'], 'o-', 
        linewidth=2, markersize=8, color='steelblue')
ax.axhline(y=0.8716, color='red', linestyle='--', linewidth=2,
           label='Baseline (87.16%)', alpha=0.7)
ax.set_xlabel('Number of PCA Components', fontsize=12, fontweight='bold')
ax.set_ylabel('CV Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Figure 5: PCA Components vs Classification Accuracy', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, 'figure5_pca_performance', report_num=1)
plt.close()

best_pca = pca_df.loc[pca_df['accuracy'].idxmax()]
print(f"\n{'='*70}")
print(f"Best PCA: n_components={best_pca['n_components']}")
print(f"Accuracy: {best_pca['accuracy']:.4f}")
print(f"{'='*70}")

# Save results
experiment_003 = {
    'experiment_id': '003',
    'description': 'PCA dimensionality reduction',
    'n_95_variance': n_95,
    'n_99_variance': n_99,
    'results_df': pca_df,
    'best_config': best_pca.to_dict(),
    'status': 'completed'
}
save_results(experiment_003, 'pca_results.pkl', 
             results_dir='results/preprocessing')

log_experiment(
    exp_num=3,
    description=f"PCA: n_components={best_pca['n_components']}",
    config={'n_components': int(best_pca['n_components']), 'scaler': best_scaler_name},
    results={'val_acc': best_pca['accuracy'], 
             'roc_auc': best_pca['roc_auc'],
             'f1_macro': best_pca['f1_score']},
    log_file='experiment_tracker.md'
)

#############################################################################
# EXPERIMENT 004: FEATURE SELECTION
#############################################################################

print("\n" + "="*70)
print("EXPERIMENT 004: FEATURE SELECTION")
print("="*70)

k_values = [50, 100, 200, 300, 400]
scoring_functions = {
    'f_classif': f_classif,
    'mutual_info': mutual_info_classif
}

fs_results = []

print("\nTesting feature selection configurations...")
for k in k_values:
    for score_name, score_func in scoring_functions.items():
        print(f"  k={k:3d}, {score_name:12s}...", end=" ")
        
        if best_scaler:
            pipeline = Pipeline([
                ('scaler', best_scaler),
                ('feature_selection', SelectKBest(score_func=score_func, k=k)),
                ('classifier', SVC(C=1.0, kernel='rbf', random_state=RANDOM_STATE))
            ])
        else:
            pipeline = Pipeline([
                ('feature_selection', SelectKBest(score_func=score_func, k=k)),
                ('classifier', SVC(C=1.0, kernel='rbf', random_state=RANDOM_STATE))
            ])
        
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=['accuracy', 'roc_auc_ovr', 'f1_macro'],
            n_jobs=-1
        )
        
        result = {
            'k': k,
            'scoring': score_name,
            'accuracy': scores['test_accuracy'].mean(),
            'std': scores['test_accuracy'].std(),
            'roc_auc': scores['test_roc_auc_ovr'].mean(),
            'f1_score': scores['test_f1_macro'].mean()
        }
        fs_results.append(result)
        print(f"Acc: {result['accuracy']:.4f}")

fs_df = pd.DataFrame(fs_results)

# Figure 6: Feature Selection Performance
fig, ax = plt.subplots(figsize=(10, 6))
for scoring in scoring_functions.keys():
    data = fs_df[fs_df['scoring'] == scoring]
    ax.plot(data['k'], data['accuracy'], marker='o', 
            linewidth=2, markersize=8, label=scoring)

ax.axhline(y=0.8716, color='red', linestyle='--', linewidth=2,
           label='Baseline (87.16%)', alpha=0.7)
ax.set_xlabel('Number of Features (k)', fontsize=12, fontweight='bold')
ax.set_ylabel('CV Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Figure 6: Feature Selection Performance', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
save_figure(fig, 'figure6_feature_selection', report_num=1)
plt.close()

best_fs = fs_df.loc[fs_df['accuracy'].idxmax()]
print(f"\n{'='*70}")
print(f"Best Feature Selection: k={best_fs['k']}, scoring={best_fs['scoring']}")
print(f"Accuracy: {best_fs['accuracy']:.4f}")
print(f"{'='*70}")

# Save results
experiment_004 = {
    'experiment_id': '004',
    'description': 'Feature selection comparison',
    'results_df': fs_df,
    'best_config': best_fs.to_dict(),
    'status': 'completed'
}
save_results(experiment_004, 'feature_selection_results.pkl', 
             results_dir='results/preprocessing')

log_experiment(
    exp_num=4,
    description=f"Feature Selection: k={best_fs['k']}, {best_fs['scoring']}",
    config={'k': int(best_fs['k']), 'scoring': best_fs['scoring'], 'scaler': best_scaler_name},
    results={'val_acc': best_fs['accuracy'], 
             'roc_auc': best_fs['roc_auc'],
             'f1_macro': best_fs['f1_score']},
    log_file='experiment_tracker.md'
)

#############################################################################
# SUMMARY
#############################################################################

print("\n" + "="*70)
print("PREPROCESSING EXPERIMENTS COMPLETE!")
print("="*70)
print(f"\nExperiment 002 (Normalization):")
print(f"  Best: {best_config['Scaler']} + {best_config['Classifier']}")
print(f"  Accuracy: {best_config['Val_Accuracy']:.4f}")

print(f"\nExperiment 003 (PCA):")
print(f"  Best: n_components={best_pca['n_components']}")
print(f"  Accuracy: {best_pca['accuracy']:.4f}")
print(f"  95% variance: {n_95} components")

print(f"\nExperiment 004 (Feature Selection):")
print(f"  Best: k={best_fs['k']}, {best_fs['scoring']}")
print(f"  Accuracy: {best_fs['accuracy']:.4f}")

print(f"\nFigures generated:")
print(f"  - Figure 3: results/figures/report1/figure3_normalization_comparison.png")
print(f"  - Figure 4: results/figures/report1/figure4_pca_variance.png")
print(f"  - Figure 5: results/figures/report1/figure5_pca_performance.png")
print(f"  - Figure 6: results/figures/report1/figure6_feature_selection.png")

print(f"\nResults saved in results/preprocessing/")
print(f"Experiment tracker updated!")
print("="*70)

