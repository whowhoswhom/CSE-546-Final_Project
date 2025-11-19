# CSE 546 Machine Learning - Final Project Task

## Project Goal
Develop a comprehensive machine learning solution for **5-class Flower Recognition** using image features, demonstrating mastery of classification techniques, ensemble methods, and analytical skills.

## Dataset
- **Training Data**: 4,065 samples × 512 features
- **Classes**: 5 flower types
  - Class 0: Daisy (757 samples, 18.6%)
  - Class 1: Dandelion (1,045 samples, 25.7%)
  - Class 2: Rose (560 samples, 13.8%)
  - Class 3: Sunflower (726 samples, 17.9%)
  - Class 4: Tulip (977 samples, 24.0%)
- **Class Imbalance**: Rose (minority) vs Dandelion (majority) = 1:1.87 ratio

## Core Tasks

### Task 1: Data Preprocessing & Analysis
- Load and explore the flower dataset
- Analyze class distribution and imbalance
- Compare 2 normalization techniques
- Implement PCA with 2 different component options
- Apply feature selection with 2 different approaches

### Task 2: Individual Classifier Development
Implement and optimize 4 different classifiers:
1. K-Nearest Neighbors (KNN)
2. Support Vector Machine (SVM)
3. Random Forest or Decision Tree
4. Multi-Layer Perceptron (MLP) or Logistic Regression

For each classifier:
- Use GridSearchCV with pipelines
- Generate learning curves
- Analyze training vs validation performance
- Identify optimal parameters with justification

### Task 3: Ensemble Methods
1. **Stacking Classifier**
   - Select diverse base models using correlation analysis
   - Test multiple meta-learners
   - Justify model selection

2. **AdaBoost**
   - Test different base estimators
   - Optimize n_estimators and learning_rate
   - Use SAMME algorithm for multiclass

### Task 4: Comprehensive Evaluation
- Use 4-fold cross-validation throughout
- Evaluate using 3 metrics: Accuracy, ROC-AUC (multiclass), F1-score
- Generate confusion matrices
- Create ROC curves for all classes
- Perform error analysis

### Task 5: Model Selection & Testing
- Select best overall model based on balanced metrics
- Prepare model for test data (to be provided Dec 3)
- Create prediction pipeline for new samples

## Deliverables

### Report 1 (Nov 21) - 50% of experiments
- Preprocessing comparisons with visualizations
- At least 2 classifiers fully optimized
- Initial results and analysis
- Plan for remaining experiments

### Final Report (Dec 5)
- All experiments completed
- Comprehensive analysis and visualizations
- Best model selection with justification
- Error analysis and insights

### Test Predictions (Dec 3)
- Apply best model to test features
- Submit CSV with predicted labels

## Success Metrics
- **Understanding**: Clear explanations of WHY methods work/fail
- **Rigor**: Systematic parameter optimization with justification
- **Analysis**: Identify overfitting, consistency across folds
- **Visualization**: Professional figures with proper numbering
- **Performance**: Competitive accuracy while avoiding overfitting
