# CSE 546 Final Project - Flower Classification

## 🌻 Multi-Class Flower Recognition using Machine Learning

A comprehensive machine learning project implementing multiple classification algorithms and ensemble methods for 5-class flower recognition using pre-extracted image features.

### 📊 Project Overview

- **Task**: 5-class flower classification
- **Dataset**: 4,065 training samples with 512 features each
- **Classes**: Daisy, Dandelion, Rose, Sunflower, Tulip
- **Challenge**: Class imbalance and high-dimensional feature space

### 🎯 Objectives

1. Compare multiple preprocessing techniques (normalization, PCA, feature selection)
2. Optimize 4 different classifier types using systematic parameter tuning
3. Implement ensemble methods (Stacking and AdaBoost)
4. Demonstrate understanding of bias-variance tradeoff and overfitting
5. Achieve competitive classification performance while maintaining generalization

### 🛠️ Methods Implemented

#### Preprocessing

- **Normalization**: StandardScaler, MinMaxScaler, RobustScaler
- **Dimensionality Reduction**: PCA with variance analysis
- **Feature Selection**: SelectKBest with multiple scoring functions

#### Classifiers

- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM) with multiple kernels
- Random Forest
- Multi-Layer Perceptron (MLP)

#### Ensemble Methods

- Stacking Classifier with correlation-based model selection
- AdaBoost with optimized base estimators

### 📈 Key Results

| Model         | CV Accuracy | ROC-AUC | F1-Score |
| ------------- | ----------- | ------- | -------- |
| Baseline KNN  | 73.5%       | 0.892   | 0.728    |
| Optimized SVM | 86.2%       | 0.941   | 0.859    |
| Best Ensemble | TBD         | TBD     | TBD      |

### 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/whowhoswhom/CSE-546-Final_ProjectV1.git
cd CSE-546-Final_ProjectV1

# Install dependencies
pip install -r requirements.txt

# Run baseline experiment
python notebooks/01_data_exploration.py
```

### 📁 Repository Structure

```
├── data/               # Dataset files
├── notebooks/          # Experiment notebooks
├── src/                # Source code modules
├── results/            # Experiment results and figures
├── models/             # Saved models
├── reports/            # Project reports
└── docs/               # Documentation
```

### 📊 Evaluation Strategy

- **Cross-Validation**: 4-fold stratified CV for all experiments
- **Metrics**: Accuracy, ROC-AUC (one-vs-rest), F1-score (macro)
- **Analysis**: Learning curves, validation curves, confusion matrices

### 🔬 Experimental Approach

1. **Systematic Parameter Optimization**: GridSearchCV with pipeline approach
2. **Overfitting Analysis**: Training vs validation score comparison
3. **Consistency Checking**: Standard deviation across CV folds
4. **Correlation Analysis**: For ensemble diversity assessment

### 📝 Documentation

- [Task Description](docs/task.md)
- [Requirements &amp; Rules](docs/rules.md)
- [Experiment Tracker](docs/experiment_tracker.md)
- [Project Setup Guide](docs/project_setup.md)

### 🏆 Key Findings

1. **Preprocessing Impact**: StandardScaler consistently improves SVM and MLP performance
2. **Optimal Dimensionality**: PCA with 100 components balances performance and efficiency
3. **Classifier Diversity**: Low correlation between tree-based and distance-based methods
4. **Ensemble Benefit**: [To be determined after final experiments]

### 📅 Timeline

- **Report 1**: November 21, 2024 (50% of experiments)
- **Test Predictions**: December 3, 2024
- **Final Submission**: December 5, 2024

### 🛠️ Technologies Used

- **Language**: Python 3.8+
- **ML Framework**: scikit-learn
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebooks, Git

### 📖 Course Information

- **Course**: CSE 546 - Introduction to Machine Learning
- **Professor**: H. Frigui
- **Institution**: University of Louisville
- **Semester**: Fall 2024

### 🤝 Acknowledgments

- Dataset source: [Flowers Recognition on Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- Course materials and guidance from Professor H. Frigui

### 📜 License

This project is for educational purposes as part of CSE 546 coursework.

---

*For detailed implementation and analysis, refer to the project reports and notebooks.*
