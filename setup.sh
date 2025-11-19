#!/bin/bash

# CSE 546 Final Project - Repository Setup Script
# This script initializes the project structure and prepares for development

echo "🚀 Setting up CSE 546 Final Project Repository..."

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create directory structure
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p data
mkdir -p docs
mkdir -p notebooks
mkdir -p src
mkdir -p results/{preprocessing,classifiers,ensemble,figures/{report1,final}}
mkdir -p models/checkpoints
mkdir -p reports/{report1,final_report}

# Create .gitkeep files to preserve empty directories
touch results/preprocessing/.gitkeep
touch results/classifiers/.gitkeep
touch results/ensemble/.gitkeep
touch results/figures/report1/.gitkeep
touch results/figures/final/.gitkeep
touch models/checkpoints/.gitkeep
touch reports/report1/.gitkeep
touch reports/final_report/.gitkeep

# Create Python package structure
touch src/__init__.py

# Create placeholder Python modules
echo "# Preprocessing functions for CSE 546 Final Project" > src/preprocessing.py
echo "# Classifier implementations" > src/classifiers.py
echo "# Evaluation metrics and plotting functions" > src/evaluation.py
echo "# Ensemble methods" > src/ensemble.py
echo "# Utility functions" > src/utils.py

# Create notebook templates
echo -e "${YELLOW}Creating notebook templates...${NC}"

cat > notebooks/01_data_exploration.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 1: Data Exploration\n",
    "## CSE 546 Final Project - Flower Classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Set random seed for reproducibility\n",
    "RANDOM_STATE = 42\n",
    "np.random.seed(RANDOM_STATE)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Copy documentation files if they exist in current directory
echo -e "${YELLOW}Copying documentation files...${NC}"
if [ -f "task.md" ]; then cp task.md docs/; fi
if [ -f "rules.md" ]; then cp rules.md docs/; fi
if [ -f "project_requirements.md" ]; then cp project_requirements.md docs/; fi
if [ -f "experiment_tracker.md" ]; then cp experiment_tracker.md docs/; fi
if [ -f "project_setup.md" ]; then cp project_setup.md docs/; fi
if [ -f "repo.md" ]; then cp repo.md ./; fi
if [ -f ".cursorrules" ]; then cp .cursorrules ./; fi
if [ -f ".gitignore" ]; then cp .gitignore ./; fi
if [ -f "requirements.txt" ]; then cp requirements.txt ./; fi
if [ -f "README.md" ]; then cp README.md ./; fi

# Copy data files if they exist
echo -e "${YELLOW}Looking for data files...${NC}"
if [ -f "flower_train_features.csv" ]; then 
    cp flower_*.csv data/
    echo -e "${GREEN}✓ Data files copied${NC}"
else
    echo "⚠️  Data files not found. Please copy them to data/ directory"
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Initializing Git repository...${NC}"
    git init
    git add README.md repo.md .gitignore requirements.txt .cursorrules
    git add docs/
    git add src/
    git add notebooks/*.ipynb
    git commit -m "Initial commit: Project structure and documentation"
    echo -e "${GREEN}✓ Git repository initialized${NC}"
else
    echo "Git repository already exists"
fi

# Create a virtual environment (optional)
echo -e "${YELLOW}Do you want to create a Python virtual environment? (y/n)${NC}"
read -r response
if [[ "$response" == "y" ]]; then
    python -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
    echo "Activate it with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
    
    # Install requirements if venv was created
    echo -e "${YELLOW}Install requirements now? (y/n)${NC}"
    read -r response2
    if [[ "$response2" == "y" ]]; then
        source venv/bin/activate
        pip install -r requirements.txt
        echo -e "${GREEN}✓ Requirements installed${NC}"
    fi
fi

# Create initial experiment log entry
cat > docs/experiment_log.md << 'EOF'
# Experiment Log - Started $(date)

## Experiment 001: Baseline
- Date: $(date +%Y-%m-%d)
- Status: Ready to start
- Configuration: KNN with k=5, no preprocessing
- Results: TBD
EOF

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Project structure created successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Copy data files to data/ directory if not already done"
echo "2. Open the project in Cursor IDE"
echo "3. Start with notebooks/01_data_exploration.ipynb"
echo "4. Track experiments in docs/experiment_tracker.md"
echo ""
echo "Repository structure ready at: $(pwd)"

# Check if remote repository needs to be added
echo ""
echo -e "${YELLOW}Add GitHub remote? (y/n)${NC}"
read -r response
if [[ "$response" == "y" ]]; then
    echo "Enter your GitHub repository URL:"
    echo "Example: https://github.com/whowhoswhom/CSE-546-Final_ProjectV1.git"
    read -r repo_url
    git remote add origin "$repo_url"
    echo -e "${GREEN}✓ Remote repository added${NC}"
    echo "Push to GitHub with: git push -u origin main"
fi

echo ""
echo "Setup complete! Happy coding! 🎉"
