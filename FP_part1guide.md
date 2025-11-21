# CSE 546 Machine Learning - Final Project Guide

## Flower Recognition Classification

### Project Overview

- **Application**: Flower Recognition (5-class classification)
- **Classes**: Daisy, Dandelion, Rose, Sunflower, Tulip
- **Dataset**: 4,065 training samples with 512 features each
- **Data Source**: Features extracted from images (original images available on Kaggle)

### Dataset Statistics

- **Total Training Samples**: 4,065
- **Feature Dimensions**: 512
- **Class Distribution**:
  - Class 0 (Daisy): 757 samples (18.6%)
  - Class 1 (Dandelion): 1,045 samples (25.7%)
  - Class 2 (Rose): 560 samples (13.8%)
  - Class 3 (Sunflower): 726 samples (17.9%)
  - Class 4 (Tulip): 977 samples (24.0%)
- **Note**: Slight class imbalance exists (Rose has fewest samples)

## Key Deliverables & Timeline

### Report 1 (Due: November 21)

- **Worth**: 20 points
- **Content**: ~50% of all experiments
- **Required**:
  - Initial experiments, results, and analysis
  - Discussion/justification of remaining experiments
- **Penalty**: 5 points off per day late

### Final Report (Due: December 5)

- **Worth**: 50 points
- **NO LATE SUBMISSIONS ACCEPTED**
- **Format Requirements**:
  - Single file (MS Word, PPT, or PDF)
  - Maximum 15 pages (12pt font)
  - All figures/tables must be numbered and referenced in text
  - Must include all figures/analysis in the report

### Audio Recording (Due: December 5)

- **Duration**: MAXIMUM 15 minutes (only first 15 min graded)
- **Structure**:
  - First 3 minutes: Live code demonstration showing execution
  - Remaining 12 minutes: Explain results, figures, comparisons
- **Requirements**: Synchronize audio with text/figures being explained

### Test Data Submission (By: December 3)

- **Worth**: 30 points
- **Process**:
  - Test features (no labels) will be provided
  - Submit CSV file with predicted labels
- **Grading Scale**:
  - Excellent: 30 pts
  - Good: 20 pts
  - Average: 10 pts
  - Non-functional/all same class: 0 pts
- **Bonus Points**:
  - 1st place: +15 points
  - 2nd place: +10 points
  - 3rd place: +5 points

### Final Submission Files (Blackboard)

1. Report (Word/PPT/PDF)
2. Link to recording
3. Notebook file
4. **DO NOT submit ZIP files**

## Technical Requirements

### Minimum Required Components

#### 1. Data Preprocessing

- **2 Normalization Options** (e.g., StandardScaler, MinMaxScaler, RobustScaler)
- **PCA with 2 Component Options** (e.g., 50 components vs 100 components, or 95% vs 99% variance)
- **Feature Selection with 2 Options** (e.g., SelectKBest with k=100 vs k=200, or different scoring functions)

#### 2. Classifiers (Minimum 4 Different Types)

Suggested options:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest or Decision Trees
- Multi-Layer Perceptron (MLP)
- Logistic Regression
- Gaussian Naive Bayes

#### 3. Ensemble Methods (Both Required)

- **Stacking Classifier**
- **AdaBoost**

#### 4. Validation Strategy

- **K-fold Cross Validation with k=4** for all validations
- **Nested 4-fold if needed** (for hyperparameter tuning)
- **Use Pipelines and GridSearch**

#### 5. Evaluation Metrics (All Required)

- **Accuracy**
- **AUC of ROC**
- **F1-measure**

### Critical Analysis Requirements

#### Parameter Selection

**You CANNOT:**

- Simply rely on GridSearch best parameters
- Select based on maximum accuracy alone
- Combine all options in one giant GridSearch
- Use algorithms not covered in class

**You MUST:**

- Plot training/validation scores
- Identify overfitting/underfitting patterns
- Analyze consistency across CV folds
- Justify parameter choices with analysis

#### Ensemble Justification

Must justify classifier selection for fusion based on:

- Individual performance metrics
- Correlation analysis of classifier outputs
- Diversity of models

### Analysis Components

#### Required Visualizations

1. **Learning Curves**: Training vs validation scores
2. **Parameter Impact Plots**: How parameters affect performance
3. **Confusion Matrices**: For best models
4. **ROC Curves**: Multi-class ROC analysis
5. **Feature Importance**: If applicable to chosen models
6. **Correlation Heatmap**: Between different classifiers' predictions

#### Required Discussions

1. Most important parameters affecting results
2. Justification of optimal parameters per classifier
3. Analysis of misclassified samples
4. Visualization of correct vs confused samples
5. Possible explanations for misclassifications

## Suggested Workflow

### Phase 1: Initial Setup & Exploration

```python
# 1. Load and explore data
- Load features, labels, filenames
- Check for missing values
- Visualize class distribution
- Compute basic statistics

# 2. Initial baseline model
- Simple train/test split
- Basic classifier without preprocessing
- Establish baseline performance
```

### Phase 2: Preprocessing Experiments

```python
# 1. Normalization comparison
- StandardScaler vs MinMaxScaler vs RobustScaler
- Impact on different classifiers

# 2. PCA analysis
- Scree plot for variance explained
- Compare 50, 100, 150 components
- Or 95% vs 99% variance retention

# 3. Feature selection
- SelectKBest with different k values
- Different scoring functions (f_classif, mutual_info_classif)
- Compare with PCA results
```

### Phase 3: Individual Classifier Optimization

```python
# For each classifier:
# 1. Initial parameter sweep
# 2. Learning curve analysis
# 3. Cross-validation consistency check
# 4. Fine-tuning based on analysis
# 5. Document optimal parameters with justification
```

### Phase 4: Ensemble Methods

```python
# 1. Correlation analysis
- Compute prediction correlations between classifiers
- Select diverse models for ensemble

# 2. Stacking
- Choose appropriate meta-learner
- Compare different base model combinations

# 3. AdaBoost
- Experiment with different base estimators
- Tune n_estimators and learning_rate
```

### Phase 5: Final Model & Analysis

```python
# 1. Train final model with best configuration
# 2. Comprehensive evaluation
# 3. Error analysis
# 4. Generate all required visualizations
```

## Code Structure Recommendations

### Use Pipelines for Clean Code

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('dim_reduction', PCA()),
    ('classifier', SVC())
])

param_grid = {
    'scaler': [StandardScaler(), MinMaxScaler()],
    'dim_reduction__n_components': [50, 100],
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=4, 
                          scoring=['accuracy', 'roc_auc_ovr', 'f1_macro'],
                          refit='accuracy', return_train_score=True)
```

### Modular Approach

```python
# Create separate functions for:
def preprocess_comparison(X, y):
    """Compare different preprocessing methods"""
    pass

def evaluate_classifier(clf, X, y, cv=4):
    """Comprehensive evaluation of a classifier"""
    pass

def plot_learning_curves(estimator, X, y):
    """Generate learning curve plots"""
    pass

def analyze_predictions(y_true, y_pred, class_names):
    """Detailed prediction analysis"""
    pass
```

## Important Notes & Tips

### DO's:

1. **Start with simple experiments** and build complexity
2. **Document every decision** with data-driven justification
3. **Keep experiments organized** - use clear naming conventions
4. **Save intermediate results** to avoid re-running long computations
5. **Use random_state** for reproducibility
6. **Consider computational time** - some combinations may take hours
7. **Focus on understanding** over pure performance

### DON'Ts:

1. **Don't use techniques not covered in class**
2. **Don't rely solely on GridSearch results**
3. **Don't select parameters based only on max accuracy**
4. **Don't combine everything in one massive GridSearch**
5. **Don't ignore class imbalance** considerations
6. **Don't forget to analyze consistency across folds**

### Professor Frigui's Emphasis:

- **Understanding > Performance**: Demonstrate why methods work/fail
- **First-person academic writing**: "I observed that..." rather than passive voice
- **Use course terminology**: Reference concepts from lectures
- **Professional presentation**: Numbered figures, clear structure
- **Thorough analysis**: Not just results, but interpretations

## Getting Started Code Template

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (cross_val_score, cross_validate, 
                                     GridSearchCV, learning_curve, 
                                     StratifiedKFold)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                             confusion_matrix, classification_report, 
                             roc_curve, auc)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier

# Load data
X_train = pd.read_csv('flower_train_features.csv', header=None)
y_train = pd.read_csv('flower_train_labels.csv', header=None).values.ravel()
filenames = pd.read_csv('flower_train_filenames.csv', header=None)
label_mapping = pd.read_csv('flower_label_mapping.csv')

print(f"Dataset shape: {X_train.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")
print(f"Class distribution:\n{pd.Series(y_train).value_counts().sort_index()}")

# Set up cross-validation
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# Continue with experiments...
```

This guide provides a complete roadmap for Part 1 of your final project. Focus on completing about 50% of these experiments for the first report due November 21st.
