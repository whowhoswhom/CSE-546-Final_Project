# CSE 546 Final Project - Action Plan & Deliverables

## Quick Start Summary

**Project**: Flower Recognition Classification (5 classes: Daisy, Dandelion, Rose, Sunflower, Tulip)
**Dataset**: 4,065 samples × 512 features
**Key Challenge**: Class imbalance (Rose: 560 samples vs Dandelion: 1,045 samples)

## Critical Deliverables & Deadlines

### 📅 November 21 - Report 1 (20 points)
**Must Include:**
- ~50% of experiments completed
- Initial preprocessing comparisons (normalization, PCA, feature selection)
- At least 2 classifiers optimized with parameter analysis
- Justification for remaining experiments
- **Format**: Report with numbered figures/tables
- **Penalty**: -5 points per day late

### 📅 December 3 - Test Data Released
**Action Required:**
- Test features will be provided (no labels)
- Submit CSV with predictions using best model
- Worth 30 points (Bonus: 1st +15, 2nd +10, 3rd +5)

### 📅 December 5 - Final Submission (50 points)
**Three Required Files:**
1. **Report** (max 15 pages, 12pt font)
2. **Recording Link** (max 15 minutes)
   - 0-3 min: Live code demo
   - 3-15 min: Results explanation
3. **Notebook** (.ipynb or .py file)

**NO ZIP FILES, NO LATE SUBMISSIONS**

## Immediate Action Items (For Report 1)

### Week 1: Foundation & Preprocessing
**Day 1-2: Setup & Baseline**
- [ ] Load data and verify dimensions
- [ ] Analyze class distribution
- [ ] Run baseline model (KNN without preprocessing)
- [ ] Document baseline accuracy

**Day 3-4: Preprocessing Experiments**
- [ ] Compare normalization methods:
  - StandardScaler vs MinMaxScaler vs RobustScaler
  - Test with KNN and SVM (scale-sensitive)
- [ ] PCA analysis:
  - Generate scree plot and cumulative variance
  - Test 50, 100, 150 components
- [ ] Feature selection:
  - SelectKBest with k=100, 200
  - Compare f_classif vs mutual_info_classif

**Day 5-7: Initial Classifiers**
- [ ] Optimize KNN:
  - k values: 3, 5, 7, 9, 11
  - Weights: uniform vs distance
  - Generate learning curves
- [ ] Optimize SVM:
  - C: 0.1, 1, 10, 100
  - Kernels: rbf, poly
  - Check overfitting

### Week 2: Complete Report 1
**Day 8-10: Additional Classifiers**
- [ ] Add one more classifier (Random Forest or MLP)
- [ ] Generate correlation matrix between classifiers
- [ ] Initial ensemble attempt (if time permits)

**Day 11-14: Analysis & Report**
- [ ] Create all required visualizations
- [ ] Write analysis with numbered figures
- [ ] Justify parameter selections
- [ ] Outline plan for remaining 50% of experiments

## Code Organization Strategy

```python
project/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_individual_classifiers.ipynb
│   ├── 04_ensemble_methods.ipynb
│   └── 05_final_analysis.ipynb
├── results/
│   ├── preprocessing_results.pkl
│   ├── classifier_results.pkl
│   └── figures/
├── models/
│   └── best_model.pkl
└── reports/
    ├── report1.docx
    └── final_report.docx
```

## Critical Success Factors

### ✅ DO's:
1. **Start simple**: Get baseline working first
2. **Save everything**: Use pickle to save results
3. **Use random_state=42**: For reproducibility
4. **Document as you go**: Don't leave writing for the end
5. **Focus on understanding**: Explain WHY not just WHAT
6. **Check fold consistency**: Look at standard deviation across CV folds

### ❌ DON'Ts:
1. **Don't trust GridSearch blindly**: Analyze the results
2. **Don't optimize everything at once**: Iterative approach
3. **Don't ignore overfitting**: Check train vs validation scores
4. **Don't use external methods**: Only course-covered techniques
5. **Don't submit ZIP files**: Individual files only

## Key Analysis Points to Address

### For Each Classifier:
1. **Parameter Impact**: How does each parameter affect performance?
2. **Overfitting Analysis**: Gap between training and validation scores
3. **Consistency Check**: Standard deviation across CV folds
4. **Optimal Selection Justification**: Why these parameters?

### For Preprocessing:
1. **Method Comparison**: Which normalization works best and why?
2. **Dimensionality Trade-off**: Performance vs computational cost
3. **Feature Importance**: Which features matter most?

### For Ensembles:
1. **Diversity Analysis**: Correlation between base classifiers
2. **Improvement Justification**: Why does ensemble help?
3. **Meta-learner Selection**: Which works best for stacking?

## Recording Tips

### Structure (15 minutes max):
```
0:00-3:00: Live code execution
- Show data loading
- Run one complete experiment
- Demonstrate output generation

3:00-7:00: Preprocessing & Individual Classifiers
- Explain normalization impact
- Show PCA/feature selection results
- Discuss parameter optimization

7:00-11:00: Ensemble Methods & Performance
- Explain ensemble strategy
- Show performance comparisons
- Analyze confusion matrices

11:00-15:00: Error Analysis & Conclusions
- Discuss misclassification patterns
- Explain key insights
- Summarize best practices learned
```

## Quick Reference Commands

```python
# Cross-validation with multiple metrics
from sklearn.model_selection import cross_validate
scores = cross_validate(pipeline, X, y, cv=4,
                        scoring=['accuracy', 'roc_auc_ovr', 'f1_macro'],
                        return_train_score=True)

# Save results for later
import joblib
joblib.dump(results_dict, 'results/experiment1.pkl')

# Generate learning curves
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    estimator, X, y, cv=4, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10))

# Confusion matrix with nice visualization
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test,
                                      display_labels=class_names,
                                      cmap='Blues')
```

## Professor Frigui's Preferences

1. **First-person writing**: "I observed that..." not passive voice
2. **Numbered figures**: "As shown in Figure 3..."
3. **Course terminology**: Use terms from lectures
4. **Justification focus**: Explain decisions with data
5. **Professional visuals**: Clear labels, titles, legends
6. **Concise code**: No verbose variable names or excessive prints

## Next Steps (Priority Order)

1. **Today**: Run the provided implementation notebook through Section 3 (Preprocessing)
2. **Tomorrow**: Complete at least 2 classifier optimizations
3. **Day 3**: Generate all visualizations and start writing
4. **Day 4-5**: Complete Report 1 and review

---

**Remember**: The goal is demonstrating understanding of ML concepts, not achieving perfect accuracy. Focus on thorough analysis and clear explanations using course concepts.

Good luck! You've got this! 🚀
