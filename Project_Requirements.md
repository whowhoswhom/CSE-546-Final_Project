# CSE 546 Machine Learning - Final Project Requirements

## Professor H. Frigui

---

## Slide 1: Overview

- **Course**: Introduction to Machine Learning
- **Assignment**: Final Project
- **Components**: Requirements, Timeline, Test and Report

---

## Slide 2: Application

### Flowers Recognition

- **Task**: Image data classification
- **Original Data**: [Kaggle Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- **Provided Data**:
  - Features are extracted and posted on Blackboard
  - Data posted can be used for training/validation
  - Test data will be made available during the demo

---

## Slide 3: Minimum Requirements

### Required Components:

1. **2 options for data normalization**
2. **PCA with 2 options for number of components (dimensions)**
3. **Feature Selection (2 options)**
4. **4 different types of classifiers**
5. **Ensemble methods: Stacking and AdaBoost**
6. **Use k-fold cross validation with k=4 for all validations**
   - Nested 4-fold if needed
7. **Use Pipelines and GridSearch**
8. **Base algorithm/parameter selection on:**
   - Accuracy
   - AUC of ROC
   - F1-measure

---

## Slide 4: Selection of Optimal Parameters for Each Classifier

### Critical Requirements:

**To select optimal parameters of each classifier:**

- ❌ You cannot rely on the best parameters identified by GridSearch
- ❌ You cannot simply select the one that yields the best accuracy

**Your analysis MUST include:**

- ✅ Plots of the training/validation scores
- ✅ Identifying potential overfitting and underfitting
- ✅ Analysis of the consistency of scores across all cross-validation folds

**Fusion Justification:**
You need to justify your choice of classifiers used in fusion based on:

- Their performance
- The correlation of their outputs

---

## Slide 5: DO NOT vs DO Guidelines

### DO NOT:

- ❌ Combine all options and parameters in one giant GridSearch!
- ❌ Use any algorithm/technique that was not covered in class
- ❌ Rely on the output of GridSearch to select optimal parameters
- ❌ Select optimal parameters based on max accuracy only!

### DO:

- ✅ Consider few options at a time
- ✅ Analyze the results
- ✅ Justify your next set of options

---

## Slide 6: Report 1

- **Due Date**: November 21
- **Late Penalty**: 5 points off per day
- **Worth**: 20 points

### Grading Based On:

- Experiments, results, and analysis
- Discussion/justification of remaining experiments
- **Must include approximately 50% of all experiments**

---

## Slide 7: Final Report

- **Due Date**: December 5 (NO LATE SUBMISSION)
- **CRITICAL**: NO REPORTS WILL BE ACCEPTED OR GRADED AFTER 12/05
- **Worth**: 50 points

### Grading Based On:

- Experiments, results, considered options, etc.
- Discussion of the most important parameters that affect the results
- Justification of the selection of optimal parameters for each classifier
- Performance of the classifier (based on cross-validation of provided data)
- Analysis of the results
- Visualization of correct samples, confused samples, etc.
- Possible justification for misclassified samples

---

## Slide 8: Final Report - What to Submit

### Report Requirements:

- **Format**: Single file (MS Word, PPT, or PDF)
- **Content**:
  - Describes your experiments
  - Summarizes, explains (using concepts covered in lectures) and compares results (using plots, tables, figures)
  - Identifies the best method
- **Length**: Cannot exceed 15 pages using font size 12
- **Figures**:
  - Assign numbers to ALL figures/tables/plots
  - Reference them in discussion using these numbers
  - Must include all figures/analysis in report
  - Can show figures during code review, but MUST include in report

---

## Slide 9: Audio Recording Requirements

### Recording Specifications:

- **Maximum Duration**: 15 minutes (only first 15 minutes will be graded)
- **Synchronization**: Must sync recording with text/figures being explained

### Time Allocation:

- **First 3 minutes**:
  - Show code is working and generating output
  - Run code while recording
- **Remaining 12 minutes**:
  - Explain results (figures)
  - Compare/explain results
  - Analysis discussion

---

## Slide 10: Submission Requirements

### Files to Submit in Blackboard:

1. Report
2. Link to the recording
3. Notebook

### CRITICAL:

**DO NOT SUBMIT a ZIP file**

---

## Slide 11: Test (of New Images)

### Test Data Process:

- **By December 3rd**: Features of test images provided (no labels, no images)
- **Task**: Test images with your best model
- **Submit**: CSV file with labels of test images
- **Scoring**: We will score these test images

### Point Distribution:

- **Maximum**: 30 points
- **Grading Scale** (based on all results including instructor's):
  - Excellent: 30 pts
  - Good: 20 pts
  - Average: 10 pts
  - Don't make sense (e.g., all assigned to same class) or code doesn't run: 0 pts

### Bonus Points:

- **1st Place**: +15 points
- **2nd Place**: +10 points
- **3rd Place**: +5 points

---

## Key Takeaways

### Timeline Summary:

1. **Nov 21**: Report 1 due (20 pts, ~50% of work)
2. **Dec 3**: Test data released, predictions due
3. **Dec 5**: Final report + recording + notebook due (50 pts)

### Critical Success Factors:

- Systematic experimentation (not all at once)
- Thorough analysis (not just accuracy)
- Justification for every decision
- Understanding overfitting/underfitting
- Professional presentation with numbered figures
- Using course concepts and terminology
