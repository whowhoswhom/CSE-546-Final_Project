# CSE 546 Report 1 - Complete Instructions

## ⏰ **DUE: November 21, 2024**

---

## 🚀 **STEP 1: RUN ALL EXPERIMENTS (1-2 hours)**

### A. Preprocessing Experiments (Exp 002-004)

```powershell
python run_all_preprocessing.py
```

**This will:**
- ✅ Test 4 normalization methods (Exp 002)
- ✅ Analyze PCA with different components (Exp 003)
- ✅ Compare feature selection methods (Exp 004)
- ✅ Generate Figures 3, 4, 5, 6
- ✅ Save results to `results/preprocessing/`
- ✅ Update `experiment_tracker.md`

**Expected runtime:** 30-45 minutes

**Expected results:**
- Normalization: StandardScaler should give ~88-89% with SVM
- PCA: ~100-150 components optimal
- Feature Selection: k=200-300 should work best

---

### B. SVM Optimization (Exp 005)

```powershell
python run_svm_optimization.py
```

**This will:**
- ✅ Optimize SVM with GridSearchCV
- ✅ Test C=[0.1, 1, 10, 100] and kernels=[rbf, poly, sigmoid]
- ✅ Generate Figures 7, 8, 9 (learning curves, parameter impact, confusion matrix)
- ✅ Save best model to `models/best_svm.pkl`
- ✅ Update `experiment_tracker.md`

**Expected runtime:** 30-60 minutes

**Expected results:**
- Best: RBF kernel, C=10 or C=100
- Accuracy: 88-90%
- Should beat baseline by 1-3%

---

### C. Commit Results

```powershell
git add results/ models/ experiment_tracker.md
git commit -m "[EXP] Preprocessing complete: StandardScaler+SVM 88.X%, SVM optimized achieves 89.X%"
git push origin main
```

---

## 📝 **STEP 2: WRITE REPORT (2-3 hours)**

### Report Structure (7-10 pages)

Create: `reports/report1/CSE546_Report1.docx` (or PDF)

---

### **1. INTRODUCTION (0.5-1 page)**

```
I present a systematic machine learning approach to 5-class flower classification 
using pre-extracted image features. Working with 4,065 samples across 512 dimensions, 
I explore preprocessing strategies and classifier optimization to maximize 
classification performance while maintaining generalization.

Research Questions:
1. Can preprocessing improve upon the strong 87.16% baseline?
2. Which preprocessing method is most effective for this dataset?
3. How much improvement can SVM optimization provide?

Methodology Overview:
- Baseline: KNN (k=5) without preprocessing
- Preprocessing: Normalization, PCA, Feature Selection
- Optimization: SVM parameter tuning with GridSearchCV
- Evaluation: 4-fold stratified cross-validation
```

---

### **2. DATASET ANALYSIS (1 page)**

**Insert Figures 1 & 2**

```
The dataset consists of 4,065 pre-extracted image features (512 dimensions) across 
5 flower classes. The class distribution shows moderate imbalance:

- Daisy: 757 samples (18.6%)
- Dandelion: 1,045 samples (25.7%) ← Majority class
- Rose: 560 samples (13.8%) ← Minority class  
- Sunflower: 726 samples (17.9%)
- Tulip: 977 samples (24.0%)

The imbalance ratio of 1.87:1 (Dandelion:Rose) is moderate and manageable with 
stratified cross-validation. As shown in Figure 1, the distribution is relatively 
balanced compared to many real-world datasets.

[Insert Table: Class Distribution with percentages]

Feature Analysis:
The 512 pre-extracted features exhibit [describe from your baseline]:
- Mean range: [X to Y]
- Standard deviation: [value]
- No missing values detected
- Features appear well-normalized based on initial exploration
```

---

### **3. BASELINE ANALYSIS (1 page)**

**Insert Figure 2**

```
I established a strong baseline using K-Nearest Neighbors (k=5) without preprocessing, 
achieving 87.16% validation accuracy—significantly exceeding the expected 70-75%. 
This exceptional performance provides several insights:

Performance Metrics:
- CV Accuracy: 87.16% (±0.54%)
- ROC-AUC: 97.27%
- F1-Score (macro): 86.61%
- Overfitting Gap: 4.58%

Key Observations:
1. Exceptional class separation: The ROC-AUC of 97.27% demonstrates that the 
   512 pre-extracted features provide excellent discriminative power.

2. Low overfitting: The gap of only 4.58% between training (91.73%) and 
   validation (87.16%) indicates good generalization.

3. Consistent performance: Standard deviation of 0.54% across folds demonstrates 
   reliable and stable results.

4. Strong baseline challenge: The high baseline means preprocessing and optimization 
   must be carefully designed to provide meaningful improvements.

As shown in Figure 2, performance was remarkably consistent across all four 
cross-validation folds, with accuracies ranging from 86.53% to 87.99%.

[Insert Table: Baseline metrics per fold]
```

---

### **4. PREPROCESSING EXPERIMENTS (2-2.5 pages)**

#### **4.1 Normalization Comparison (Experiment 002)**

**Insert Figure 3**

```
Given the strong baseline, I systematically compared four normalization approaches:
- None (raw features)
- StandardScaler (zero mean, unit variance)
- MinMaxScaler (0-1 range)
- RobustScaler (median and IQR)

I tested each with both KNN (k=7) and SVM (C=1.0, RBF kernel) to understand 
how normalization affects different classifier families.

Results (from Figure 3):
[Insert your actual results table here]

| Scaler | KNN Acc | SVM Acc |
|--------|---------|---------|
| None | X.XX% | X.XX% |
| StandardScaler | X.XX% | X.XX% |
| MinMaxScaler | X.XX% | X.XX% |
| RobustScaler | X.XX% | X.XX% |

Key Findings:
1. StandardScaler provided the best performance with SVM, achieving X.XX% 
   (improvement of +X.XX% over baseline).

2. KNN was less sensitive to normalization, as expected for distance-based 
   methods that rely on local neighborhoods.

3. SVM showed significant improvement with StandardScaler because [explain: 
   SVM relies on inner products, standardization helps convergence, etc.]

4. The overfitting gap remained controlled (<5%) across all configurations.

Best Configuration: StandardScaler + SVM (C=1.0, RBF)
- This becomes the baseline for subsequent experiments
```

#### **4.2 PCA Analysis (Experiment 003)**

**Insert Figures 4a, 4b, 5**

```
To understand dimensionality reduction potential, I performed comprehensive PCA 
analysis on the standardized features.

Variance Analysis:
- 95% variance retained: XXX components (Figure 4b)
- 99% variance retained: XXX components
- First 50 components: XX.X% variance (Figure 4a)

I tested classification performance with different component counts using the 
best preprocessing (StandardScaler + SVM).

Results (Figure 5):
[Insert table]

| n_components | Accuracy | vs Baseline |
|--------------|----------|-------------|
| 50 | X.XX% | -X.XX% |
| 100 | X.XX% | -X.XX% |
| 150 | X.XX% | +/-X.XX% |
| XXX (95%) | X.XX% | +/-X.XX% |

Key Findings:
1. Dimensionality reduction from 512 to ~100-150 components maintained 
   XX.X% accuracy, showing that substantial redundancy exists.

2. [Interpretation: Did accuracy drop significantly? If so, explain that 
   some dimensions beyond 95% variance contain discriminative information.]

3. Computational benefit: Reducing to 100 components decreases training 
   time by ~XX% while maintaining X.XX% of performance.

4. Trade-off: While PCA provides computational efficiency, it slightly 
   [improves/reduces] accuracy compared to using all features.

Optimal Configuration: n_components=XXX achieves best accuracy-efficiency balance.
```

#### **4.3 Feature Selection (Experiment 004)**

**Insert Figure 6**

```
As an alternative to PCA, I evaluated univariate feature selection using SelectKBest 
with two scoring functions:
- f_classif: ANOVA F-statistic
- mutual_info_classif: Mutual information

Results (Figure 6):
[Insert table]

| k | f_classif | mutual_info |
|---|-----------|-------------|
| 50 | X.XX% | X.XX% |
| 100 | X.XX% | X.XX% |
| 200 | X.XX% | X.XX% |
| 300 | X.XX% | X.XX% |
| 400 | X.XX% | X.XX% |

Key Findings:
1. [Which scoring function worked better and why?]

2. Optimal k=[XXX] features provided X.XX% accuracy.

3. Comparison with PCA:
   - Feature selection [outperformed/underperformed] PCA by X.XX%
   - [Explain: Feature selection maintains interpretability, PCA creates 
     new features that are combinations]

4. Both methods demonstrate that the 512-dimensional space contains 
   significant redundancy.

Best Configuration: SelectKBest with k=XXX, [scoring function]
```

---

### **5. SVM OPTIMIZATION (2 pages)**

**Insert Figures 7, 8, 9**

```
Building on the best preprocessing (StandardScaler), I systematically optimized 
SVM using GridSearchCV with 4-fold cross-validation.

Parameter Space:
- C: [0.1, 1, 10, 100]
- Kernel: ['rbf', 'poly', 'sigmoid']
- Gamma: ['scale', 'auto']

Total configurations tested: 24

Optimization Results:
Best Parameters:
- Kernel: [XXX]
- C: [XXX]
- Gamma: [XXX]

Performance:
- Training Accuracy: X.XX%
- Validation Accuracy: X.XX%
- ROC-AUC: X.XX%
- F1-Score: X.XX%
- Overfitting Gap: X.XX%
- Fold Std Dev: X.XX%

Improvement: +X.XX% over 87.16% baseline

Analysis (Figure 7 - Learning Curves):
The learning curves reveal [describe what you see]:
1. [Does training accuracy plateau? What does this mean?]
2. [Is there significant gap between train and val? Overfitting?]
3. [Do curves converge? Would more data help?]

Parameter Impact (Figure 8):
The C parameter analysis shows:
1. [What happened as C increased? Did validation accuracy plateau?]
2. [Which C value provided best bias-variance trade-off?]
3. [Explain why this C value works: regularization balance]

Kernel Comparison:
[Compare all three kernels]:
- RBF: X.XX% - [Why it worked/didn't work]
- Poly: X.XX% - [Analysis]
- Sigmoid: X.XX% - [Analysis]

Confusion Matrix Analysis (Figure 9):
The confusion matrix reveals interesting misclassification patterns:
1. [Which classes are most confused? Why?]
2. [Example: "Rose and Tulip show XX misclassifications, likely due to 
   similar color features"]
3. [Any surprising results? Classes that classify perfectly?]

Cross-Validation Consistency:
Fold-by-fold accuracy: [X.XX%, X.XX%, X.XX%, X.XX%]
Standard deviation: X.XX%
[Interpret: Is std < 0.01 excellent? < 0.02 good?]
```

---

### **6. RESULTS SUMMARY (1 page)**

**Insert Comparison Table**

```
| Experiment | Configuration | CV Acc | ROC-AUC | F1-Score | Overfit Gap |
|------------|--------------|---------|---------|----------|-------------|
| 001 Baseline | KNN k=5, no prep | 87.16% | 97.27% | 86.61% | 4.58% |
| 002 Best Norm | StandardScaler + SVM | X.XX% | X.XX% | X.XX% | X.XX% |
| 003 Best PCA | n_comp=XXX | X.XX% | X.XX% | X.XX% | X.XX% |
| 004 Best FS | k=XXX, f_classif | X.XX% | X.XX% | X.XX% | X.XX% |
| 005 SVM Opt | [kernel], C=[X] | X.XX% | X.XX% | X.XX% | X.XX% |

Progress Summary:
- Baseline → Normalization: +X.XX% improvement
- Baseline → SVM Optimized: +X.XX% total improvement
- Final accuracy: X.XX%

Key Achievement: [Did you break 90%? If yes, highlight. If no, explain why 
                   exceptional baseline makes further gains challenging]

Statistical Significance:
[Discuss whether improvements are meaningful given the standard deviations]
```

---

### **7. KEY INSIGHTS (0.5-1 page)**

```
Through systematic experimentation, I gained several important insights:

1. Feature Quality Matters Most:
   The exceptional 87.16% baseline demonstrates that the pre-extracted features 
   are highly discriminative. This suggests that [discuss feature extraction 
   quality, domain knowledge in feature engineering].

2. Preprocessing Provides Modest but Meaningful Gains:
   StandardScaler improved SVM performance by X.XX%, demonstrating that proper 
   normalization is crucial for kernel-based methods. However, the relatively 
   small improvement indicates the features were already well-scaled.

3. Dimensionality Reduction Trade-offs:
   Both PCA and feature selection showed that ~100-300 features contain most 
   discriminative information. This has practical implications for [computational 
   efficiency, deployment, interpretability].

4. Optimization Challenges with Strong Baselines:
   When baseline performance is already high (>85%), incremental improvements 
   require careful tuning. The law of diminishing returns applies—each additional 
   percentage point becomes harder to achieve.

5. Overfitting Remains Well-Controlled:
   Across all experiments, overfitting gaps remained below 5%, indicating good 
   generalization. This suggests the 4,065 samples provide adequate training data.

6. Kernel Selection:
   [Your kernel] proved most effective because [explain: RBF captures non-linear 
   relationships, polynomial might overfit, sigmoid might be too simple].

Limitations Observed:
- Class imbalance (Rose minority) may affect performance on that class
- Computational cost of GridSearchCV with 24 configurations
- [Any other challenges you noticed]
```

---

### **8. FUTURE WORK (0.5 page)**

```
For the final report (December 5), I will complete the remaining ~50% of experiments:

Planned Experiments:

1. Additional Classifier Optimization:
   - Random Forest: Test n_estimators, max_depth, min_samples_split
   - MLP: Optimize hidden layers, activation functions, learning rate
   - Expected: Each should achieve 88-90% accuracy

2. Ensemble Methods (Required):
   
   a) Stacking Classifier:
      - Combine diverse base models (KNN, SVM, RF, MLP)
      - Use correlation analysis to select models with low correlation
      - Test different meta-learners (Logistic Regression, SVM, RF)
      - Expected: 90-92% accuracy through model diversity
   
   b) AdaBoost:
      - Test with Decision Tree base estimators of varying depths
      - Optimize n_estimators and learning_rate
      - Use SAMME algorithm for multiclass
      - Expected: 89-91% accuracy, good handling of difficult samples

3. Comprehensive Analysis:
   - Generate classifier correlation heatmap
   - Analyze which samples benefit most from ensembling
   - Error analysis: Why do some flowers consistently misclassify?
   - Feature importance analysis across all methods

4. Final Model Selection:
   - Select best model based on accuracy, generalization, and complexity
   - Prepare for test data prediction (December 3)
   - Target: >92% accuracy on test set

Timeline:
- Week of Nov 22-28: RF and MLP optimization
- Week of Nov 29-Dec 3: Ensemble methods
- Dec 3: Test predictions submission
- Dec 4-5: Final report compilation
```

---

## 📋 **FORMATTING CHECKLIST**

Before submission, verify:

- [ ] All 9 figures embedded and numbered (Figure 1-9)
- [ ] All figures referenced in text ("As shown in Figure X...")
- [ ] First-person academic writing throughout
- [ ] All metrics in tables use consistent decimal places (4 decimal: 0.XXXX or 2 decimal: XX.XX%)
- [ ] Page count: 7-10 pages (check!)
- [ ] Font: 12pt
- [ ] All tables have titles/captions
- [ ] File format: PDF or Word (NO ZIP!)
- [ ] Filename: CSE546_Report1_[YourName].pdf

---

## 🎯 **WRITING TIPS**

### Good Example:
```
"I observed that StandardScaler improved SVM accuracy by 1.2% (from 87.16% to 88.36%) 
because standardization ensures all features contribute equally to the kernel 
computation. Without standardization, features with larger magnitudes dominate 
the distance calculations, potentially reducing model performance."
```

### Bad Example:
```
"StandardScaler was tested and got better results."
```

### Key Principles:
1. **Always explain WHY, not just WHAT**
2. **Use specific numbers**: "improved by 1.2%" not "improved slightly"
3. **Reference figures**: "As shown in Figure 3" not "the chart shows"
4. **First-person**: "I found" not "it was found"
5. **Connect to concepts**: Mention bias-variance, overfitting, generalization

---

## ⏰ **TIMELINE FOR COMPLETION**

**Today (Nov 20):**
- [2 hours] Run all experiments
- [1 hour] Analyze results, take notes
- [2 hours] Write sections 1-4

**Tomorrow Morning (Nov 21):**
- [1.5 hours] Write sections 5-8
- [30 mins] Format, proofread
- [30 mins] Generate PDF, final check
- [ ] **SUBMIT by deadline**

---

## 🆘 **TROUBLESHOOTING**

### If scripts take too long:
- Reduce parameter grid in `run_svm_optimization.py`
- Use `n_jobs=-1` (should already be set)
- Run on a faster machine if available

### If accuracy doesn't improve:
- That's OK! Explain in report why baseline is hard to beat
- Focus on analysis and understanding
- Strong baseline + good analysis > weak baseline + high improvement

### If figures look bad:
- Increase DPI in `save_figure()` function
- Use `plt.tight_layout()` before saving
- Ensure labels are visible and fonts are readable

---

**Good luck! Your exceptional baseline gives you a great story to tell.** 🚀

