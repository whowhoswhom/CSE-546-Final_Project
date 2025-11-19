# CSE 546 Final Project - Repository Overview
## Flower Classification with Machine Learning

**Repository**: https://github.com/whowhoswhom/CSE-546-Final_ProjectV1  
**Course**: CSE 546 - Introduction to Machine Learning  
**Professor**: H. Frigui  
**Semester**: Fall 2024  

---

## 🎯 Project Mission
Develop a robust 5-class flower classification system using traditional machine learning techniques, demonstrating deep understanding of preprocessing, optimization, and ensemble methods through systematic experimentation and analysis.

---

## 📊 Dataset Overview
- **Training Samples**: 4,065 images (pre-extracted features)
- **Feature Dimensions**: 512
- **Classes**: 5 flower types
  ```
  0: Daisy     (757 samples, 18.6%)
  1: Dandelion (1,045 samples, 25.7%)
  2: Rose      (560 samples, 13.8%)  ← Minority class
  3: Sunflower (726 samples, 17.9%)
  4: Tulip     (977 samples, 24.0%)
  ```
- **Challenge**: Class imbalance ratio 1.87:1 (Dandelion:Rose)

---

## 📁 Repository Structure

```
CSE-546-Final_ProjectV1/
│
├── README.md                 # Public repository description
├── repo.md                   # This file - Internal project context
├── .cursorrules             # Cursor AI context and constraints
├── .gitignore               # Git ignore rules
│
├── data/                    # Dataset files (git-ignored if large)
│   ├── flower_train_features.csv
│   ├── flower_train_labels.csv
│   ├── flower_train_filenames.csv
│   ├── flower_label_mapping.csv
│   └── test_features.csv    # (Added Dec 3)
│
├── docs/                    # Project documentation
│   ├── task.md             # Project objectives
│   ├── rules.md            # Requirements and constraints
│   ├── project_requirements.md  # Original assignment
│   ├── experiment_tracker.md    # Experiment logging
│   └── project_setup.md    # Setup guide
│
├── notebooks/               # Jupyter notebooks (numbered sequence)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_experiments.ipynb
│   ├── 03_individual_classifiers.ipynb
│   ├── 04_ensemble_methods.ipynb
│   ├── 05_final_model_selection.ipynb
│   └── 06_test_predictions.ipynb
│
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py    # Preprocessing functions
│   ├── classifiers.py      # Classifier implementations
│   ├── evaluation.py       # Evaluation and plotting
│   ├── ensemble.py         # Ensemble methods
│   └── utils.py           # Utility functions
│
├── results/                 # Experiment results and outputs
│   ├── preprocessing/      # Preprocessing experiments
│   ├── classifiers/        # Individual classifier results
│   ├── ensemble/          # Ensemble method results
│   └── figures/           # All figures for reports
│       ├── report1/       # Figures for Report 1
│       └── final/         # Figures for Final Report
│
├── models/                  # Saved models
│   ├── checkpoints/        # Intermediate model saves
│   ├── best_model.pkl      # Final selected model
│   └── model_comparison.json  # Performance comparison
│
├── reports/                 # Report materials
│   ├── report1/
│   │   ├── report1.docx
│   │   └── figures/
│   └── final_report/
│       ├── final_report.docx
│       ├── recording_link.txt
│       └── figures/
│
└── requirements.txt         # Python dependencies
```

---

## 🔄 Git Workflow & Version Control

### Branch Strategy
```
main (protected)
├── development (active work)
├── feature/preprocessing
├── feature/classifiers
├── feature/ensemble
├── report/report1
└── report/final
```

### Commit Convention
```bash
# Format: [TYPE] Component: Description

[FEAT] Preprocessing: Add PCA variance analysis
[FIX] KNN: Correct cross-validation scoring
[DOC] Report1: Add learning curve analysis
[EXP] SVM: Test polynomial kernel with C=10
[PLOT] Ensemble: Generate correlation heatmap
```

### Types:
- `[FEAT]` - New feature/functionality
- `[FIX]` - Bug fix
- `[DOC]` - Documentation/report updates
- `[EXP]` - Experiment (include results in message)
- `[PLOT]` - Figure/visualization generation
- `[OPT]` - Optimization/performance improvement
- `[TEST]` - Test data predictions

### Tagging Milestones
```bash
git tag -a baseline-complete -m "Baseline KNN: 73.5% accuracy"
git tag -a report1-submission -m "Report 1 submitted: Nov 21"
git tag -a best-model-v1 -m "Best model: SVM RBF C=10, 89.3%"
git tag -a final-submission -m "Final submission: Dec 5"
```

---

## 📈 Development Status

### Current Phase: **Preprocessing & Initial Classifiers**

#### Progress Tracker
```
Overall Progress: ██████░░░░ 60%

✅ Completed:
├── [x] Project setup and documentation
├── [x] Data exploration and visualization
├── [x] Baseline model (KNN, no preprocessing)
├── [x] Normalization comparison
└── [x] PCA analysis

🔄 In Progress:
├── [ ] Feature selection experiments
├── [ ] KNN full optimization
└── [ ] SVM parameter tuning

📋 Upcoming:
├── [ ] Random Forest optimization
├── [ ] MLP implementation
├── [ ] Ensemble methods (Stacking, AdaBoost)
└── [ ] Final model selection
```

### Key Metrics Dashboard
| Metric | Baseline | Current Best | Target | Status |
|--------|----------|--------------|--------|---------|
| CV Accuracy | 73.5% | 86.2% | 90%+ | 🔄 |
| ROC-AUC | 0.892 | 0.941 | 0.95+ | 🔄 |
| F1-Macro | 0.728 | 0.859 | 0.88+ | 🔄 |
| Overfitting Gap | 15.3% | 3.2% | <5% | ✅ |

---

## 🧪 Experiment Registry

### Best Configurations Found
```python
# Best Preprocessing
Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100))
])

# Best KNN
KNeighborsClassifier(
    n_neighbors=7,
    weights='distance',
    metric='euclidean'
)

# Best SVM (pending)
SVC(
    C=10,
    kernel='rbf',
    gamma='scale',
    probability=True
)
```

### Failed Experiments (Learning Points)
- ❌ PCA with 50 components: Too much information loss
- ❌ Polynomial kernel degree>3: Overfitting
- ❌ No scaling with SVM: Poor convergence

---

## 📝 Key Deadlines & Deliverables

| Date | Deliverable | Status | Points |
|------|------------|---------|---------|
| Nov 21 | Report 1 (50% experiments) | 🔄 In Progress | 20 pts |
| Dec 3 | Test predictions | ⏳ Waiting | 30 pts |
| Dec 5 | Final report | ⏳ Waiting | 50 pts |
| Dec 5 | Recording (<15 min) | ⏳ Waiting | - |
| Dec 5 | Notebook submission | ⏳ Waiting | - |

---

## 🛠️ Development Guidelines

### For Every New Experiment
1. Create new branch: `git checkout -b exp/experiment-name`
2. Update `experiment_tracker.md` with configuration
3. Run experiment and save results to `results/`
4. Generate required plots to `results/figures/`
5. Commit with descriptive message including key metrics
6. Merge to development if successful

### Before Each Commit
- [ ] Code runs without errors
- [ ] Results saved to appropriate directory
- [ ] Experiment logged in tracker
- [ ] Figures generated and numbered
- [ ] Documentation updated if needed

### Code Quality Checklist
- [ ] Functions have docstrings
- [ ] Complex operations commented
- [ ] Random state set to 42
- [ ] Pipeline approach used
- [ ] Cross-validation with k=4
- [ ] Multiple metrics evaluated

---

## 🚀 Quick Commands

### Setup Environment
```bash
# Clone repository
git clone https://github.com/whowhoswhom/CSE-546-Final_ProjectV1.git
cd CSE-546-Final_ProjectV1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments
```python
# Standard experiment execution
python -m notebooks.02_preprocessing_experiments
python -m notebooks.03_individual_classifiers

# Generate all figures for report
python src/evaluation.py --generate-all-figures --output results/figures/report1/
```

### Git Operations
```bash
# Start new experiment
git checkout development
git pull origin development
git checkout -b exp/adaboost-optimization

# Save experiment results
git add results/ notebooks/
git commit -m "[EXP] AdaBoost: Best config n_est=100, lr=0.5, acc=87.3%"

# Prepare for submission
git checkout report/report1
git merge development
git tag -a report1-ready -m "Report 1 ready for submission"
```

---

## 📊 Results Summary (Auto-Updated)

### Latest Experiment Results
| Exp# | Date | Method | Best Config | CV Acc | Status |
|------|------|--------|-------------|---------|---------|
| 001 | 11/15 | Baseline KNN | k=5, no scaling | 73.5% | ✅ |
| 002 | 11/16 | KNN + StandardScaler | k=7 | 81.2% | ✅ |
| 003 | 11/16 | KNN + PCA | k=7, n=100 | 83.4% | ✅ |
| 004 | 11/17 | SVM RBF | C=1, scale | 86.2% | ✅ |
| 005 | 11/18 | Feature Selection | k=200, f_classif | 82.1% | ✅ |
| ... | ... | ... | ... | ... | ... |

### Model Performance Ranking
1. **SVM (RBF)**: 86.2% accuracy, 0.941 ROC-AUC
2. **KNN (optimized)**: 83.4% accuracy, 0.918 ROC-AUC
3. **Random Forest**: [Pending]
4. **MLP**: [Pending]

---

## 🔗 Important Links

- **Course Materials**: [Blackboard/Canvas link]
- **Original Dataset**: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
- **Repository**: https://github.com/whowhoswhom/CSE-546-Final_ProjectV1
- **Previous Homeworks Reference**: [HW1-5 solutions in /mnt/project/]

---

## 💡 Context for Cursor AI

### When Working on This Project:
1. **Always check** `docs/rules.md` for requirements
2. **Reference** `docs/experiment_tracker.md` for experiment history
3. **Follow** patterns from successful experiments
4. **Use** 4-fold CV and pipelines consistently
5. **Generate** numbered figures for all results
6. **Document** decisions and justifications
7. **Focus on** understanding over raw performance

### Key Constraints:
- Only scikit-learn (no deep learning)
- 4-fold cross-validation mandatory
- Must use pipelines
- All 3 metrics required (accuracy, ROC-AUC, F1)
- Maximum 15 pages for final report
- Recording maximum 15 minutes

### Professor's Priorities:
1. Systematic experimentation
2. Clear justifications for choices
3. Understanding of overfitting/underfitting
4. Professional presentation
5. Use of course concepts

---

## 📧 Contact & Collaboration

**Project Owner**: Toni (Jose Fuentes)  
**Course**: CSE 546 - Introduction to Machine Learning  
**Institution**: University of Louisville  
**Semester**: Fall 2024  

---

## 📝 Notes Section

### Current Focus
- Completing preprocessing experiments for Report 1
- Optimizing KNN and SVM thoroughly
- Preparing learning curve visualizations

### Blockers/Issues
- None currently

### Next Steps
1. Complete feature selection comparison
2. Finalize KNN optimization
3. Start Random Forest implementation
4. Begin Report 1 writing

---

*Last Updated: November 2024*  
*Auto-sync with experiment_tracker.md for latest results*
