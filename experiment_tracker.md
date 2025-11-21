# Experiment Tracking Sheet - Final Project

## Quick Summary Table
| Exp# | Date | Method | Best Params | CV Acc | Train Acc | Overfit | Time | Status |
|------|------|--------|-------------|--------|-----------|---------|------|--------|
| 001  |      | Baseline KNN | k=5, no preprocessing | | | | | ✅ |
| 002  |      | KNN + StandardScaler | | | | | | ⏳ |
| 003  |      | KNN + MinMaxScaler | | | | | | ⏳ |
| ... |      | | | | | | | |

---

## Detailed Experiment Logs

### EXPERIMENT 001: Baseline
**Date**: [Date]  
**Objective**: Establish baseline performance without preprocessing

**Configuration**:
```python
model = KNeighborsClassifier(n_neighbors=5)
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
```

**Results**:
- CV Accuracy: X.XX ± X.XX
- Training Accuracy: X.XX
- Per-fold scores: [X.XX, X.XX, X.XX, X.XX]
- Execution time: XX seconds

**Observations**:
- [Key insight 1]
- [Key insight 2]

**Next Steps**:
- Test with normalization
- Try different k values

---

### EXPERIMENT 002: Normalization Comparison - KNN
**Date**: [Date]  
**Objective**: Compare normalization impact on KNN

**Configuration**:
```python
scalers = [None, StandardScaler(), MinMaxScaler(), RobustScaler()]
k_values = [3, 5, 7, 9]
```

**Results Table**:
| Scaler | k | CV Acc | Train Acc | Overfit Gap | Best |
|--------|---|--------|-----------|-------------|------|
| None | 5 | | | | |
| Standard | 5 | | | | ✓ |
| MinMax | 5 | | | | |
| Robust | 5 | | | | |

**Key Finding**: [Which scaler works best and why]

**Figure Generated**: `Figure_002_normalization_comparison.png`

---

### EXPERIMENT 003: PCA Analysis
**Date**: [Date]  
**Objective**: Determine optimal number of PCA components

**Configuration**:
```python
n_components_options = [50, 100, 150, 200, 0.95, 0.99]
base_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', SVC())
])
```

**Results**:
| n_components | Actual Dims | Variance | CV Acc | Time(s) |
|--------------|-------------|----------|--------|---------|
| 50 | 50 | XX% | | |
| 100 | 100 | XX% | | |
| 150 | 150 | XX% | | |
| 0.95 | XXX | 95% | | |
| 0.99 | XXX | 99% | | |

**Observations**:
- Knee in scree plot at component XX
- 95% variance achieved with XXX components
- Optimal trade-off at XXX components

**Figures Generated**: 
- `Figure_003a_scree_plot.png`
- `Figure_003b_cumulative_variance.png`

---

### EXPERIMENT 004: Feature Selection
**Date**: [Date]  
**Objective**: Compare feature selection with PCA

**Configuration**:
```python
k_values = [50, 100, 200, 300, 400]
score_funcs = [f_classif, mutual_info_classif]
```

**Results**:
| Method | k | Score Func | CV Acc | vs PCA |
|--------|---|------------|--------|---------|
| SelectKBest | 100 | f_classif | | |
| SelectKBest | 100 | mutual_info | | |
| SelectKBest | 200 | f_classif | | |
| PCA | 100 | - | | baseline |

**Key Finding**: [PCA vs Feature Selection comparison]

---

### EXPERIMENT 005: KNN Optimization
**Date**: [Date]  
**Objective**: Full KNN parameter optimization

**Parameter Grid**:
```python
param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}
```

**Best Configuration**:
- n_neighbors: XX
- weights: XXXX
- metric: XXXX
- CV Score: X.XXX

**Overfitting Analysis**:
- Training score: X.XXX
- Validation score: X.XXX
- Gap: X.XXX (indicates [overfitting/good fit/underfitting])

**Fold Consistency**:
- Fold scores: [X.XX, X.XX, X.XX, X.XX]
- Standard deviation: X.XXX
- Conclusion: [Stable/Unstable] across folds

**Figures Generated**:
- `Figure_005a_knn_learning_curve.png`
- `Figure_005b_knn_validation_curve.png`

---

### EXPERIMENT 006: SVM Optimization
**Date**: [Date]  
**Objective**: Optimize SVM with different kernels

**Parameter Grid**:
```python
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['rbf', 'poly', 'sigmoid'],
    'classifier__gamma': ['scale', 'auto', 0.001, 0.01]
}
```

**Results Summary**:
| Kernel | Best C | Best γ | CV Acc | Train Acc | Gap |
|--------|--------|--------|--------|-----------|-----|
| RBF | | | | | |
| Poly | | | | | |
| Sigmoid | | | | | |

**Best Configuration**: [Details]

**Analysis**: [Why this configuration works best]

---

## Preprocessing Decision Matrix

| Classifier | Best Scaler | PCA Components | Feature Selection | Final Choice |
|------------|-------------|----------------|-------------------|--------------|
| KNN | | | | |
| SVM | | | | |
| RF | | | | |
| MLP | | | | |

## Classifier Performance Summary

| Classifier | Best CV Acc | ROC-AUC | F1-Macro | Overfit? | Selected for Ensemble? |
|------------|-------------|---------|----------|----------|------------------------|
| KNN | | | | | |
| SVM | | | | | |
| RF | | | | | |
| MLP | | | | | |

## Ensemble Experiments

### EXPERIMENT E01: Classifier Correlation
**Date**: [Date]  
**Correlation Matrix**:
```
        KNN    SVM    RF    MLP
KNN     1.00   X.XX   X.XX  X.XX
SVM     X.XX   1.00   X.XX  X.XX
RF      X.XX   X.XX   1.00  X.XX
MLP     X.XX   X.XX   X.XX  1.00
```

**Selected for Stacking**: [List based on diversity]

### EXPERIMENT E02: Stacking
**Meta-learners Tested**:
| Meta-learner | Base Models | CV Acc | Improvement |
|--------------|-------------|--------|-------------|
| LogisticReg | KNN, SVM, RF | | |
| DecisionTree | KNN, SVM, RF | | |
| SVM-Linear | KNN, RF, MLP | | |

### EXPERIMENT E03: AdaBoost
**Configurations Tested**:
| Base | n_estimators | learning_rate | CV Acc |
|------|--------------|---------------|--------|
| DT(d=1) | 50 | 1.0 | |
| DT(d=2) | 100 | 0.5 | |
| DT(d=3) | 200 | 0.1 | |

---

## Final Model Selection

**Selected Model**: [Name]  
**Configuration**: [Full details]  
**Final Metrics**:
- Accuracy: X.XXX
- ROC-AUC: X.XXX  
- F1-Score: X.XXX

**Justification**:
1. [Reason 1]
2. [Reason 2]
3. [Reason 3]

---

## Test Set Predictions

**Date Submitted**: Dec 3, 2024  
**Predictions File**: `test_predictions.csv`  
**Model Used**: [Exact configuration]  
**Expected Performance**: ~X.XX based on CV

---

## Lessons Learned

### What Worked Well:
1. 
2. 
3. 

### What Didn't Work:
1. 
2. 
3. 

### Surprising Findings:
1. 
2. 

### For Future Projects:
1. 
2. 

---

## Time Log

| Task | Planned Hours | Actual Hours | Notes |
|------|---------------|--------------|-------|
| Data Exploration | 2 | | |
| Preprocessing | 4 | | |
| KNN Optimization | 3 | | |
| SVM Optimization | 3 | | |
| RF Optimization | 3 | | |
| MLP Optimization | 3 | | |
| Ensemble Methods | 4 | | |
| Report Writing | 6 | | |
| Recording | 2 | | |
| **TOTAL** | 30 | | |

---

## Report Checklist

### Report 1 (Nov 21)
- [ ] Cover page with name, date, course
- [ ] Abstract/Introduction
- [ ] Data description with class distribution
- [ ] Preprocessing experiments (2 normalizations, PCA, feature selection)
- [ ] 2+ classifier optimizations with full analysis
- [ ] Learning curves and parameter impact plots
- [ ] Overfitting analysis
- [ ] Fold consistency analysis
- [ ] All figures numbered and referenced
- [ ] Plan for remaining 50% of experiments
- [ ] 7-10 pages total

### Final Report (Dec 5)
- [ ] All 4 classifiers complete
- [ ] Ensemble methods implemented
- [ ] Correlation analysis
- [ ] Confusion matrices
- [ ] ROC curves (multiclass)
- [ ] Error analysis with examples
- [ ] Final model selection justified
- [ ] Test set approach described
- [ ] Conclusions and insights
- [ ] ≤15 pages total

### Recording (Dec 5)
- [ ] Test recording setup
- [ ] Prepare script/outline
- [ ] 0-3 min: Live code demo
- [ ] 3-7 min: Preprocessing and individual classifiers
- [ ] 7-11 min: Ensemble methods and results
- [ ] 11-15 min: Analysis and conclusions
- [ ] Upload and get shareable link
- [ ] Test link before submission

---

## Notes Section

[Space for additional observations, ideas, or reminders]

## Experiment 001
- **Date**: 2025-11-20 18:58:04
- **Description**: Baseline KNN (k=5) without preprocessing
- **Configuration**: {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'euclidean', 'preprocessing': None}
- **Results**:
  - Accuracy: 0.8716
  - ROC-AUC: 0.9727
  - F1-Score: 0.8661
- **Status**: Completed

---
