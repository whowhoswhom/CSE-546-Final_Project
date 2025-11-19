import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import (
    cross_val_score, cross_validate, GridSearchCV, 
    learning_curve, StratifiedKFold, validation_curve
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, 
    confusion_matrix, classification_report, 
    roc_curve, auc, make_scorer
)
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# For multiclass ROC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

print("All libraries imported successfully!")

#%% [markdown]
# # CSE 546: Machine Learning Final Project
# ## Flower Recognition Classification
# 
# **Student**: [Your Name]  
# **Date**: November 2024  
# **Dataset**: Flower Recognition (5 classes)

#%% [markdown]
# ## 1. Data Loading and Initial Exploration

#%%
# Load the data
X_train = pd.read_csv('../uploads/flower_train_features.csv', header=None)
y_train = pd.read_csv('../uploads/flower_train_labels.csv', header=None).values.ravel()
filenames = pd.read_csv('../uploads/flower_train_filenames.csv', header=None)
label_mapping = pd.read_csv('../uploads/flower_label_mapping.csv')

# Display basic information
print(f"Dataset shape: {X_train.shape}")
print(f"Number of samples: {X_train.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of classes: {len(np.unique(y_train))}")
print("\nLabel mapping:")
print(label_mapping)

#%%
# Class distribution analysis
class_counts = pd.Series(y_train).value_counts().sort_index()
class_names = label_mapping['class_name'].values

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar plot
axes[0].bar(class_counts.index, class_counts.values)
axes[0].set_xticks(class_counts.index)
axes[0].set_xticklabels(class_names, rotation=45)
axes[0].set_xlabel('Flower Class')
axes[0].set_ylabel('Number of Samples')
axes[0].set_title('Class Distribution')

# Add count labels on bars
for i, (idx, count) in enumerate(class_counts.items()):
    axes[0].text(idx, count + 10, str(count), ha='center')

# Pie chart
axes[1].pie(class_counts.values, labels=class_names, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Class Distribution (%)')

plt.suptitle('Figure 1: Dataset Class Distribution Analysis')
plt.tight_layout()
plt.show()

# Print imbalance analysis
print("\nClass Distribution Analysis:")
for idx, name in enumerate(class_names):
    count = class_counts[idx]
    percentage = (count / len(y_train)) * 100
    print(f"Class {idx} ({name:10s}): {count:4d} samples ({percentage:5.1f}%)")

max_class = class_counts.max()
min_class = class_counts.min()
print(f"\nImbalance Ratio: {max_class/min_class:.2f} (max/min)")
print(f"Most represented: {class_names[class_counts.idxmax()]} ({max_class} samples)")
print(f"Least represented: {class_names[class_counts.idxmin()]} ({min_class} samples)")

#%%
# Check for missing values and basic statistics
print("Missing values check:")
print(f"Features: {X_train.isnull().sum().sum()} missing values")
print(f"Labels: {pd.Series(y_train).isnull().sum()} missing values")

print("\nFeature statistics:")
print(X_train.describe().loc[['mean', 'std', 'min', 'max']])

#%%
# Visualize feature distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Sample first few features for visualization
sample_features = [0, 10, 50, 100]
for ax, feat_idx in zip(axes.flat, sample_features):
    for class_idx in range(5):
        class_data = X_train.iloc[y_train == class_idx, feat_idx]
        ax.hist(class_data, alpha=0.5, label=class_names[class_idx], bins=20)
    ax.set_xlabel(f'Feature {feat_idx}')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_title(f'Distribution of Feature {feat_idx}')

plt.suptitle('Figure 2: Sample Feature Distributions Across Classes')
plt.tight_layout()
plt.show()

#%% [markdown]
# ## 2. Baseline Model (No Preprocessing)

#%%
# Establish baseline with simple KNN
from sklearn.model_selection import cross_val_score

# Set up cross-validation
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# Simple KNN baseline
knn_baseline = KNeighborsClassifier(n_neighbors=5)
baseline_scores = cross_val_score(knn_baseline, X_train, y_train, cv=cv, scoring='accuracy')

print("Baseline KNN (k=5, no preprocessing):")
print(f"Accuracy per fold: {baseline_scores}")
print(f"Mean accuracy: {baseline_scores.mean():.4f} (+/- {baseline_scores.std() * 2:.4f})")

#%% [markdown]
# ## 3. Data Preprocessing Experiments
# 
# ### 3.1 Normalization Comparison

#%%
# Compare different scalers
scalers = {
    'None': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

# Test with KNN (sensitive to scale) and SVC
classifiers_to_test = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVC': SVC(kernel='rbf', C=1.0, random_state=42)
}

normalization_results = []

for scaler_name, scaler in scalers.items():
    for clf_name, clf in classifiers_to_test.items():
        if scaler is None:
            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
        else:
            pipeline = Pipeline([
                ('scaler', scaler),
                ('classifier', clf)
            ])
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        
        normalization_results.append({
            'Scaler': scaler_name,
            'Classifier': clf_name,
            'Mean_Accuracy': scores.mean(),
            'Std': scores.std(),
            'Scores': scores
        })
        print(f"{scaler_name:15s} + {clf_name:5s}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

#%%
# Visualize normalization comparison
norm_df = pd.DataFrame(normalization_results)

fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35
x = np.arange(len(scalers))

for i, clf_name in enumerate(classifiers_to_test.keys()):
    clf_data = norm_df[norm_df['Classifier'] == clf_name]
    offset = width * i - width/2
    ax.bar(x + offset, clf_data['Mean_Accuracy'], width, 
           label=clf_name, yerr=clf_data['Std'])

ax.set_xlabel('Normalization Method')
ax.set_ylabel('Accuracy')
ax.set_title('Figure 3: Impact of Normalization on Different Classifiers')
ax.set_xticks(x)
ax.set_xticklabels(scalers.keys())
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Analysis
best_scaler = norm_df.loc[norm_df['Mean_Accuracy'].idxmax()]
print(f"\nBest configuration: {best_scaler['Scaler']} with {best_scaler['Classifier']}")
print(f"Accuracy: {best_scaler['Mean_Accuracy']:.4f}")

#%% [markdown]
# ### 3.2 PCA Analysis

#%%
# PCA variance explained analysis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca_full = PCA()
pca_full.fit(X_scaled)

# Calculate cumulative variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find components for different variance thresholds
var_thresholds = [0.90, 0.95, 0.99]
n_components_for_var = []
for threshold in var_thresholds:
    n_comp = np.argmax(cumulative_variance >= threshold) + 1
    n_components_for_var.append(n_comp)
    print(f"Components for {threshold*100:.0f}% variance: {n_comp}")

#%%
# Visualize PCA variance
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scree plot
axes[0].plot(range(1, 51), pca_full.explained_variance_ratio_[:50], 'bo-')
axes[0].set_xlabel('Component Number')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Scree Plot (First 50 Components)')
axes[0].grid(alpha=0.3)

# Cumulative variance
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
axes[1].axhline(y=0.90, color='r', linestyle='--', label='90% variance')
axes[1].axhline(y=0.95, color='g', linestyle='--', label='95% variance')
axes[1].axhline(y=0.99, color='b', linestyle='--', label='99% variance')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Figure 4: PCA Variance Analysis')
plt.tight_layout()
plt.show()

#%%
# Compare different PCA component options
pca_options = [50, 100, 150, n_components_for_var[1]]  # Including 95% variance
pca_results = []

for n_comp in pca_options:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_comp, random_state=42)),
        ('classifier', SVC(kernel='rbf', C=1.0, random_state=42))
    ])
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    pca_results.append({
        'n_components': n_comp,
        'accuracy': scores.mean(),
        'std': scores.std()
    })
    
    print(f"PCA({n_comp:3d} components): {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

#%% [markdown]
# ### 3.3 Feature Selection

#%%
# Feature selection comparison
k_values = [50, 100, 200, 300]
scoring_functions = {'f_classif': f_classif, 'mutual_info': mutual_info_classif}
fs_results = []

for k in k_values:
    for score_name, score_func in scoring_functions.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(score_func=score_func, k=k)),
            ('classifier', SVC(kernel='rbf', C=1.0, random_state=42))
        ])
        
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        fs_results.append({
            'k': k,
            'scoring': score_name,
            'accuracy': scores.mean(),
            'std': scores.std()
        })
        
        print(f"SelectKBest(k={k:3d}, {score_name:12s}): {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

#%%
# Visualize feature selection results
fs_df = pd.DataFrame(fs_results)

fig, ax = plt.subplots(figsize=(10, 6))
for scoring in scoring_functions.keys():
    data = fs_df[fs_df['scoring'] == scoring]
    ax.errorbar(data['k'], data['accuracy'], yerr=data['std'], 
                marker='o', label=scoring, capsize=5)

ax.set_xlabel('Number of Features (k)')
ax.set_ylabel('Accuracy')
ax.set_title('Figure 5: Feature Selection Performance')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%% [markdown]
# ## 4. Individual Classifier Optimization
# 
# ### 4.1 K-Nearest Neighbors (KNN)

#%%
def evaluate_classifier_detailed(pipeline, param_grid, X, y, cv, clf_name="Classifier"):
    """Comprehensive evaluation of a classifier with GridSearch"""
    
    # Perform GridSearch
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv,
        scoring=['accuracy', 'roc_auc_ovr', 'f1_macro'],
        refit='accuracy',
        return_train_score=True,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    # Extract results
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Best parameters
    print(f"\n{clf_name} - Best Parameters (by accuracy):")
    print(grid_search.best_params_)
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
    
    # Check consistency across folds
    best_idx = grid_search.best_index_
    fold_scores = []
    for i in range(cv.n_splits):
        fold_scores.append(results_df.iloc[best_idx][f'split{i}_test_accuracy'])
    
    print(f"Fold scores: {fold_scores}")
    print(f"Std across folds: {np.std(fold_scores):.4f}")
    
    return grid_search, results_df

#%%
# KNN optimization
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),
    ('classifier', KNeighborsClassifier())
])

knn_param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 15],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

knn_grid, knn_results = evaluate_classifier_detailed(
    knn_pipeline, knn_param_grid, X_train, y_train, cv, "KNN"
)

#%%
# Visualize KNN parameter impact
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# K value impact
k_uniform = knn_results[
    (knn_results['param_classifier__weights'] == 'uniform') & 
    (knn_results['param_classifier__metric'] == 'euclidean')
]
k_distance = knn_results[
    (knn_results['param_classifier__weights'] == 'distance') & 
    (knn_results['param_classifier__metric'] == 'euclidean')
]

axes[0].plot(k_uniform['param_classifier__n_neighbors'], 
             k_uniform['mean_test_accuracy'], 'o-', label='uniform')
axes[0].plot(k_distance['param_classifier__n_neighbors'], 
             k_distance['mean_test_accuracy'], 's-', label='distance')
axes[0].set_xlabel('K (number of neighbors)')
axes[0].set_ylabel('CV Accuracy')
axes[0].set_title('Impact of K and Weights')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Train vs validation scores
axes[1].plot(k_uniform['param_classifier__n_neighbors'], 
             k_uniform['mean_train_accuracy'], 'o-', label='Train (uniform)')
axes[1].plot(k_uniform['param_classifier__n_neighbors'], 
             k_uniform['mean_test_accuracy'], 'o--', label='Validation (uniform)')
axes[1].set_xlabel('K (number of neighbors)')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training vs Validation Scores')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Figure 6: KNN Parameter Analysis')
plt.tight_layout()
plt.show()

# Overfitting analysis
best_k = knn_grid.best_params_['classifier__n_neighbors']
print(f"\nOverfitting Analysis for KNN:")
best_row = knn_results.iloc[knn_grid.best_index_]
train_score = best_row['mean_train_accuracy']
val_score = best_row['mean_test_accuracy']
print(f"Best K={best_k}: Train={train_score:.4f}, Val={val_score:.4f}")
print(f"Overfitting gap: {(train_score - val_score):.4f}")

#%% [markdown]
# ### 4.2 Support Vector Machine (SVM)

#%%
# SVM optimization
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),
    ('classifier', SVC(random_state=42, probability=True))
])

svm_param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['rbf', 'poly', 'sigmoid'],
    'classifier__gamma': ['scale', 'auto']
}

svm_grid, svm_results = evaluate_classifier_detailed(
    svm_pipeline, svm_param_grid, X_train, y_train, cv, "SVM"
)

#%%
# Learning curves for SVM
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv, title="Learning Curve"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='orange')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, val_mean, 's-', color='orange', label='Validation score')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.show()

# Plot learning curve for best SVM
best_svm = svm_grid.best_estimator_
plot_learning_curve(best_svm, X_train, y_train, cv, 
                    "Figure 7: SVM Learning Curve (Best Parameters)")

#%% [markdown]
# ### 4.3 Random Forest

#%%
# Random Forest optimization
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Use fewer parameter combinations for initial exploration
rf_param_grid_small = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [10, None],
    'classifier__min_samples_split': [2, 5]
}

rf_grid, rf_results = evaluate_classifier_detailed(
    rf_pipeline, rf_param_grid_small, X_train, y_train, cv, "Random Forest"
)

#%% [markdown]
# ### 4.4 Multi-Layer Perceptron (MLP)

#%%
# MLP optimization
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),
    ('classifier', MLPClassifier(random_state=42, max_iter=1000))
])

mlp_param_grid = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
    'classifier__alpha': [0.0001, 0.001, 0.01],
    'classifier__learning_rate': ['constant', 'adaptive']
}

mlp_grid, mlp_results = evaluate_classifier_detailed(
    mlp_pipeline, mlp_param_grid, X_train, y_train, cv, "MLP"
)

#%% [markdown]
# ## 5. Classifier Correlation Analysis

#%%
# Train optimized classifiers and get predictions
from sklearn.model_selection import cross_val_predict

# Get predictions from each optimized classifier
classifiers = {
    'KNN': knn_grid.best_estimator_,
    'SVM': svm_grid.best_estimator_,
    'RF': rf_grid.best_estimator_,
    'MLP': mlp_grid.best_estimator_
}

predictions = {}
for name, clf in classifiers.items():
    preds = cross_val_predict(clf, X_train, y_train, cv=cv)
    predictions[name] = preds
    accuracy = accuracy_score(y_train, preds)
    print(f"{name} CV Accuracy: {accuracy:.4f}")

#%%
# Compute correlation matrix between classifier predictions
pred_df = pd.DataFrame(predictions)
correlation_matrix = pred_df.corr()

# Visualize correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0.5,
            vmin=0, vmax=1, square=True, linewidths=1)
plt.title('Figure 8: Classifier Prediction Correlation Matrix')
plt.tight_layout()
plt.show()

# Analysis
print("\nCorrelation Analysis:")
for i in range(len(correlation_matrix)):
    for j in range(i+1, len(correlation_matrix)):
        corr = correlation_matrix.iloc[i, j]
        name1 = correlation_matrix.index[i]
        name2 = correlation_matrix.columns[j]
        print(f"{name1} vs {name2}: {corr:.4f}")

#%% [markdown]
# ## 6. Ensemble Methods
# 
# ### 6.1 Stacking Classifier

#%%
# Based on correlation analysis, select diverse classifiers for stacking
# Choose classifiers with lower correlation for better diversity

base_estimators = [
    ('knn', knn_grid.best_estimator_),
    ('svm', svm_grid.best_estimator_),
    ('rf', rf_grid.best_estimator_)
]

# Try different meta-learners
meta_learners = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'SVM': SVC(kernel='linear', random_state=42, probability=True)
}

stacking_results = []
for meta_name, meta_clf in meta_learners.items():
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_clf,
        cv=cv  # Use same CV strategy for stacking
    )
    
    scores = cross_validate(stacking_clf, X_train, y_train, cv=cv,
                           scoring=['accuracy', 'roc_auc_ovr', 'f1_macro'])
    
    stacking_results.append({
        'Meta-learner': meta_name,
        'Accuracy': scores['test_accuracy'].mean(),
        'ROC-AUC': scores['test_roc_auc_ovr'].mean(),
        'F1': scores['test_f1_macro'].mean(),
        'Acc_std': scores['test_accuracy'].std()
    })
    
    print(f"Stacking with {meta_name:20s}: Acc={scores['test_accuracy'].mean():.4f}, "
          f"AUC={scores['test_roc_auc_ovr'].mean():.4f}, "
          f"F1={scores['test_f1_macro'].mean():.4f}")

#%% [markdown]
# ### 6.2 AdaBoost

#%%
# AdaBoost with different base estimators
base_estimators_ada = {
    'Decision Tree (depth=1)': DecisionTreeClassifier(max_depth=1, random_state=42),
    'Decision Tree (depth=2)': DecisionTreeClassifier(max_depth=2, random_state=42),
    'Decision Tree (depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42)
}

n_estimators_options = [50, 100, 200]
learning_rates = [0.1, 0.5, 1.0]

ada_results = []
for base_name, base_est in base_estimators_ada.items():
    for n_est in n_estimators_options:
        for lr in learning_rates:
            ada_clf = AdaBoostClassifier(
                base_estimator=base_est,
                n_estimators=n_est,
                learning_rate=lr,
                random_state=42,
                algorithm='SAMME'  # For multiclass
            )
            
            # Wrap in pipeline with preprocessing
            ada_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=100)),
                ('classifier', ada_clf)
            ])
            
            scores = cross_val_score(ada_pipeline, X_train, y_train, cv=cv, scoring='accuracy')
            
            ada_results.append({
                'Base': base_name,
                'n_estimators': n_est,
                'learning_rate': lr,
                'accuracy': scores.mean(),
                'std': scores.std()
            })

# Find best AdaBoost configuration
ada_df = pd.DataFrame(ada_results)
best_ada = ada_df.loc[ada_df['accuracy'].idxmax()]
print("\nBest AdaBoost Configuration:")
print(best_ada)

#%%
# Visualize AdaBoost parameter impact
fig, ax = plt.subplots(figsize=(10, 6))

for base in base_estimators_ada.keys():
    data = ada_df[(ada_df['Base'] == base) & (ada_df['learning_rate'] == 1.0)]
    ax.plot(data['n_estimators'], data['accuracy'], marker='o', label=base)

ax.set_xlabel('Number of Estimators')
ax.set_ylabel('CV Accuracy')
ax.set_title('Figure 9: AdaBoost Performance vs Number of Estimators')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%% [markdown]
# ## 7. Final Model Selection and Evaluation

#%%
# Compare all models
all_models = {
    'KNN (optimized)': knn_grid.best_estimator_,
    'SVM (optimized)': svm_grid.best_estimator_,
    'RF (optimized)': rf_grid.best_estimator_,
    'MLP (optimized)': mlp_grid.best_estimator_,
    'Stacking': StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=cv
    ),
    'AdaBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),
        ('classifier', AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=int(best_ada['Base'].split('=')[1][0]), random_state=42),
            n_estimators=int(best_ada['n_estimators']),
            learning_rate=best_ada['learning_rate'],
            random_state=42,
            algorithm='SAMME'
        ))
    ])
}

# Comprehensive evaluation
final_results = []
for name, model in all_models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv,
                           scoring=['accuracy', 'roc_auc_ovr', 'f1_macro'])
    
    final_results.append({
        'Model': name,
        'Accuracy': scores['test_accuracy'].mean(),
        'Acc_std': scores['test_accuracy'].std(),
        'ROC-AUC': scores['test_roc_auc_ovr'].mean(),
        'F1-macro': scores['test_f1_macro'].mean()
    })

final_df = pd.DataFrame(final_results).sort_values('Accuracy', ascending=False)
print("\nFinal Model Comparison:")
print(final_df.to_string())

#%%
# Visualize final comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['Accuracy', 'ROC-AUC', 'F1-macro']
for ax, metric in zip(axes, metrics):
    ax.barh(final_df['Model'], final_df[metric])
    ax.set_xlabel(metric)
    ax.set_title(f'Model Comparison - {metric}')
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Figure 10: Final Model Performance Comparison')
plt.tight_layout()
plt.show()

#%% [markdown]
# ## 8. Best Model Analysis

#%%
# Select best model based on balanced performance
best_model_name = final_df.iloc[0]['Model']
best_model = all_models[best_model_name]
print(f"Selected Best Model: {best_model_name}")

# Get predictions for confusion matrix
y_pred = cross_val_predict(best_model, X_train, y_train, cv=cv)

# Confusion Matrix
cm = confusion_matrix(y_train, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Figure 11: Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_train, y_pred, target_names=class_names))

#%%
# ROC Curves for multiclass
# Train model on full training set for ROC analysis
best_model.fit(X_train, y_train)

# Binarize the labels for multiclass ROC
y_train_bin = label_binarize(y_train, classes=[0, 1, 2, 3, 4])
n_classes = y_train_bin.shape[1]

# Get prediction probabilities
if hasattr(best_model, 'predict_proba'):
    y_score = cross_val_predict(best_model, X_train, y_train, cv=cv, method='predict_proba')
else:
    # For models without predict_proba, use decision_function
    y_score = cross_val_predict(best_model, X_train, y_train, cv=cv, method='decision_function')

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_train_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Figure 12: ROC Curves for All Classes')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%% [markdown]
# ## 9. Error Analysis

#%%
# Find misclassified samples
misclassified_indices = np.where(y_train != y_pred)[0]
print(f"Number of misclassified samples: {len(misclassified_indices)}")

# Analyze misclassification patterns
misclass_df = pd.DataFrame({
    'True': y_train[misclassified_indices],
    'Predicted': y_pred[misclassified_indices],
    'Index': misclassified_indices
})

# Count misclassification pairs
misclass_counts = misclass_df.groupby(['True', 'Predicted']).size().reset_index(name='Count')
misclass_counts['True_Name'] = misclass_counts['True'].map(lambda x: class_names[x])
misclass_counts['Predicted_Name'] = misclass_counts['Predicted'].map(lambda x: class_names[x])
misclass_counts = misclass_counts.sort_values('Count', ascending=False)

print("\nTop 10 Misclassification Patterns:")
print(misclass_counts[['True_Name', 'Predicted_Name', 'Count']].head(10))

#%%
# Visualize misclassification patterns
pivot_table = pd.pivot_table(misclass_df, values='Index', index='True', 
                             columns='Predicted', aggfunc='count', fill_value=0)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted (for errors only)')
plt.ylabel('True (for errors only)')
plt.title('Figure 13: Misclassification Heatmap')
plt.tight_layout()
plt.show()

#%% [markdown]
# ## 10. Summary and Conclusions
# 
# ### Key Findings:
# 
# 1. **Data Preprocessing Impact:**
#    - StandardScaler consistently improved performance
#    - PCA with 100 components balanced dimensionality reduction with information retention
#    - Feature selection showed [results to be filled]
# 
# 2. **Individual Classifier Performance:**
#    - Best KNN: [parameters and performance]
#    - Best SVM: [parameters and performance]
#    - Best RF: [parameters and performance]
#    - Best MLP: [parameters and performance]
# 
# 3. **Ensemble Methods:**
#    - Stacking improved individual classifier performance by [X%]
#    - AdaBoost with [parameters] achieved [performance]
# 
# 4. **Error Analysis:**
#    - Most confusion between [classes]
#    - Possible reasons: [analysis]
# 
# ### Recommendations for Part 2:
# 1. [Recommendation 1]
# 2. [Recommendation 2]
# 3. [Recommendation 3]

#%%
# Save the best model for testing
import joblib

# Save the best model
joblib.dump(best_model, 'best_model_part1.pkl')
print(f"Best model saved as 'best_model_part1.pkl'")

# Function to load and test on new data
def test_on_new_data(test_features_path, model_path='best_model_part1.pkl'):
    """Function to test on new data when it becomes available"""
    # Load model
    model = joblib.load(model_path)
    
    # Load test features
    X_test = pd.read_csv(test_features_path, header=None)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Save predictions
    pd.DataFrame(y_pred, columns=['predicted_label']).to_csv('test_predictions.csv', index=False)
    print("Predictions saved to 'test_predictions.csv'")
    
    return y_pred

print("\nReady for Part 2 of the project!")
