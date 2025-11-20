# Running Baseline Experiment 001

## ✅ Pre-Flight Check Passed!
All imports verified, data loading works correctly:
- Data shape: (4065, 512) ✓
- Classes: 5 flower types ✓
- Modules: All functional ✓

---

## 🚀 Method 1: Run in Jupyter Notebook (Recommended)

### Step 1: Start Jupyter
```powershell
cd notebooks
jupyter notebook
```

### Step 2: Open Notebook
- Browser will open automatically
- Click on `01_data_exploration.ipynb`

### Step 3: Execute All Cells
- Click: **Kernel** → **Restart & Run All**
- Or: Press **Shift + Enter** through each cell

### Step 4: Wait for Completion
- Expected runtime: 2-5 minutes
- Watch for "Experiment 001 completed successfully!" message

---

## 🖥️ Method 2: Command Line Execution

```powershell
# Convert and execute notebook
jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb --ExecutePreprocessor.timeout=300
```

---

## 📊 Expected Results

### Console Output Should Show:
```
Dataset Information:
============================================================
Feature matrix shape: (4065, 512)
Number of samples: 4065
Number of features: 512
Number of classes: 5

Baseline Results:
============================================================
Training Accuracy:   0.XXXX (±0.XXXX)
Validation Accuracy: 0.7XXX (±0.0XXX)  <- Target: 70-75%
ROC-AUC (OvR):       0.8XXX              <- Target: 0.88-0.92
F1-Score (macro):    0.7XXX              <- Target: 0.70-0.75
Overfitting Gap:     0.XXXX              <- Should be reasonable
```

### Files Created:
1. ✅ `results/preprocessing/baseline_results.pkl`
2. ✅ `results/figures/report1/figure1_class_distribution.png`
3. ✅ `results/figures/report1/figure2_baseline_performance.png`
4. ✅ `experiment_tracker.md` (updated with Experiment 001)

---

## 🔍 Post-Execution Verification

### Check 1: Verify Results File
```powershell
python -c "import joblib; r = joblib.load('results/preprocessing/baseline_results.pkl'); print(f'Accuracy: {r[\"results\"][\"val_acc\"]:.4f}')"
```

### Check 2: Verify Figures
```powershell
dir results\figures\report1\
# Should show: figure1_class_distribution.png, figure2_baseline_performance.png
```

### Check 3: Check Experiment Tracker
```powershell
Get-Content experiment_tracker.md -Tail 20
# Should show Experiment 001 entry
```

---

## 📝 After Successful Run: Commit Results

```powershell
# Check what changed
git status

# Stage the results
git add results/preprocessing/baseline_results.pkl
git add results/figures/report1/
git add experiment_tracker.md
git add notebooks/01_data_exploration.ipynb

# Commit with results (replace X.XXXX with actual values)
git commit -m "[EXP] Baseline: KNN k=5 no preprocessing, CV acc=X.XXXX, ROC-AUC=0.XXXX, F1=0.XXXX"

# Push to GitHub
git push origin main
```

---

## ⚠️ Troubleshooting

### Issue: ModuleNotFoundError
**Solution:**
```python
# Add this in first cell if needed
import sys
sys.path.append('..')
```

### Issue: Figures not saving
**Solution:**
```python
import os
os.makedirs('../results/figures/report1', exist_ok=True)
```

### Issue: Kernel dies during execution
**Solution:**
- Check RAM usage (high-dimensional data)
- Try running cells individually
- Restart kernel and try again

### Issue: Results differ significantly from expected
**If accuracy < 65%:**
- Check data loading (should be 4065 samples)
- Verify labels are correct [0, 1, 2, 3, 4]
- Check for data corruption

**If accuracy > 80%:**
- Verify no data leakage
- Check CV is properly configured
- Ensure no preprocessing applied

---

## 🎯 Next Steps After Baseline Success

### Immediate (Same Session):
1. ✅ Verify all outputs created
2. ✅ Review figures quality
3. ✅ Commit and push results
4. ✅ Tag this milestone:
   ```powershell
   git tag -a baseline-v1 -m "Baseline KNN: XX.X% accuracy"
   git push origin --tags
   ```

### Next Session (Preprocessing Experiments):
Create `notebooks/02_preprocessing_experiments.ipynb`:

**Experiment 002: Normalization Comparison**
- StandardScaler, MinMaxScaler, RobustScaler
- Test with KNN and SVM
- Expected improvement: +5-10% accuracy

**Experiment 003: PCA Analysis**
- Test n_components: 50, 100, 150
- Generate scree plot (Figure 3)
- Cumulative variance analysis

**Experiment 004: Feature Selection**
- SelectKBest with k=100, 200, 300
- Compare f_classif vs mutual_info_classif

---

## 📈 Baseline Performance Context

### Why 70-75% is Expected:
1. **No preprocessing** - Raw features
2. **Simple KNN** - k=5 with uniform weights
3. **Class imbalance** - Rose has 560 samples, Dandelion has 1,045
4. **High dimensionality** - 512 features without reduction

### This Baseline Provides:
- ✅ Lower bound for improvement
- ✅ Validation that data is correct
- ✅ CV strategy verification
- ✅ Evaluation pipeline confirmation

### Expected Improvements Later:
- **With StandardScaler**: +5-8% accuracy
- **With PCA (100 components)**: +3-5% accuracy
- **With optimized parameters**: +5-10% accuracy
- **With ensemble methods**: +3-5% accuracy
- **Total potential**: 85-90% accuracy

---

## 📊 Report 1 Progress Tracker

```
Report 1 Components (Due Nov 21):
├── [✓] Setup & Documentation
├── [✓] Data Exploration
├── [→] Baseline Experiment (ready to run!)
├── [ ] Preprocessing Comparison (2-3 experiments)
├── [ ] Classifier 1: KNN Optimization
├── [ ] Classifier 2: SVM Optimization
└── [ ] Report Writing & Figures

Progress: ████░░░░░░ 40% → 50% (after baseline)
```

---

## 🎓 Key Takeaways

### What This Baseline Establishes:
1. **Data pipeline works** - Loading, splitting, evaluation
2. **CV strategy correct** - StratifiedKFold with 4 folds
3. **Metrics calculation** - All 3 required metrics
4. **Figure generation** - Numbered, publication-quality
5. **Result logging** - Automatic tracking

### What You'll Learn:
- Baseline performance without preprocessing
- Class distribution impact on accuracy
- Per-fold consistency (std < 0.02 is good)
- Overfitting indicators (train vs validation gap)

---

**Status**: 🟢 Ready to Execute  
**Confidence**: High  
**Risk**: Minimal

**Execute the notebook now and watch your first ML experiment run!** 🚀

