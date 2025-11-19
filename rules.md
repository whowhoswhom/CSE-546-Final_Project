# Final Project Rules and Requirements

## Mandatory Technical Requirements

### Preprocessing (ALL required)
1. **2 Normalization Options**
   - Examples: StandardScaler, MinMaxScaler, RobustScaler, Normalizer
   - Compare impact on different classifiers
   - Document which works best for each algorithm

2. **PCA with 2 Component Options**
   - Option 1: Fixed number (e.g., 50, 100, 150 components)
   - Option 2: Variance-based (e.g., 95%, 99% variance retained)
   - Include scree plot and cumulative variance analysis

3. **Feature Selection with 2 Options**
   - Different k values (e.g., k=100, k=200, k=300)
   - Different scoring functions (f_classif, mutual_info_classif)
   - Compare with PCA results

### Classifiers (Minimum 4 different types)
- Each classifier must be optimized separately
- Cannot combine all in one giant GridSearch
- Must use Pipeline approach
- Required: systematic parameter sweeps with analysis

### Ensemble Methods (BOTH required)
1. **Stacking Classifier**
   - Must justify base model selection
   - Test different meta-learners
   - Use correlation analysis for diversity

2. **AdaBoost**
   - Use SAMME algorithm for multiclass
   - Optimize base estimator parameters
   - Test different n_estimators and learning rates

### Validation Strategy
- **4-fold Cross Validation** for ALL experiments
- **Nested cross-validation** if needed for hyperparameter tuning
- Must use same CV split strategy throughout for consistency

### Evaluation Metrics (ALL required)
- **Accuracy**: Overall classification accuracy
- **ROC-AUC**: Use 'ovr' (one-vs-rest) for multiclass
- **F1-score**: Use 'macro' averaging

## Critical Analysis Requirements

### Parameter Selection Rules
**YOU CANNOT:**
- ❌ Rely solely on GridSearchCV best_params_
- ❌ Select parameters based only on maximum accuracy
- ❌ Use any algorithm NOT covered in class
- ❌ Combine all options in one massive GridSearch

**YOU MUST:**
- ✅ Plot training vs validation scores
- ✅ Identify and discuss overfitting/underfitting
- ✅ Analyze consistency across CV folds (check std deviation)
- ✅ Justify EVERY parameter choice with data/plots
- ✅ Consider computational efficiency

### Ensemble Justification Requirements
1. **Correlation Analysis**
   - Compute correlation matrix between classifier predictions
   - Select models with lower correlation for diversity
   - Include correlation heatmap visualization

2. **Performance Justification**
   - Compare individual vs ensemble performance
   - Explain why ensemble improves (or doesn't)
   - Analyze which samples benefit from ensemble

## Report Requirements

### Format Rules
- **Maximum 15 pages** (12pt font)
- **Single file**: Word, PPT, or PDF (NO ZIP FILES)
- **All figures must be numbered**: "Figure 1:", "Figure 2:", etc.
- **Reference figures in text**: "As shown in Figure 3..."
- **First-person academic writing**: "I observed..." not passive voice
- **Use course terminology**: Reference concepts from lectures

### Required Visualizations
1. Class distribution (bar chart and/or pie chart)
2. Normalization comparison plot
3. PCA scree plot and cumulative variance
4. Feature selection performance comparison
5. Learning curves for each classifier
6. Parameter impact plots (training vs validation)
7. Cross-validation fold consistency analysis
8. Classifier correlation heatmap
9. Confusion matrices for best models
10. Multiclass ROC curves
11. Error analysis visualizations

### Required Discussions
1. Most important parameters affecting results
2. Justification of optimal parameters for EACH classifier
3. Analysis of overfitting/underfitting for each model
4. Consistency analysis across CV folds
5. Performance comparison across all models
6. Misclassification pattern analysis
7. Possible explanations for errors

## Recording Requirements
- **Maximum 15 minutes** (only first 15 min graded)
- **Structure**:
  - 0-3 min: Live code execution demonstration
  - 3-15 min: Results explanation and analysis
- **Must synchronize** audio with figures/text being discussed
- **Must show code actually running** in first 3 minutes

## Submission Requirements

### Files to Submit (Blackboard)
1. Report (Word/PPT/PDF) - NO ZIP
2. Link to recording
3. Notebook (.ipynb or .py file)

### Test Predictions (Dec 3)
- CSV file with predicted labels
- Column name: 'predicted_label'
- One prediction per row
- No header except column name

## Grading Focus

### What Professor Values Most
1. **Understanding > Raw Performance**
   - Explain WHY methods work or fail
   - Use concepts from lectures
   - Show deep comprehension

2. **Systematic Approach**
   - Start simple, build complexity
   - Document decision process
   - Show iterative improvement

3. **Professional Presentation**
   - Clear, numbered figures
   - Organized structure
   - Concise, focused writing

### Common Mistakes to Avoid
- Don't use techniques not covered in class
- Don't submit results without analysis
- Don't ignore overfitting indicators
- Don't select parameters without justification
- Don't forget to analyze fold consistency
- Don't combine everything in one GridSearch
- Don't use external libraries beyond scikit-learn basics

## Code Requirements

### Must Use
- scikit-learn pipelines
- GridSearchCV or RandomizedSearchCV
- cross_validate for multiple metrics
- StratifiedKFold for balanced splits
- random_state=42 for reproducibility

### Best Practices
- Modular code with functions
- Clear variable names
- Comments explaining decisions
- Save intermediate results
- Efficient computation (avoid redundancy)

## Timeline Penalties
- **Report 1**: -5 points per day late
- **Final Report**: NO LATE SUBMISSIONS ACCEPTED
- **Test Predictions**: Must submit by deadline for scoring

## Academic Integrity
- All code must be your own
- Can use course materials and standard scikit-learn
- Must cite any external references
- Collaboration discussion OK, but individual implementation
