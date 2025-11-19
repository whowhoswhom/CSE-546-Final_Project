"""
Preprocessing functions for CSE 546 Final Project
Includes normalization, PCA, and feature selection utilities
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


def load_data(data_path='data/'):
    """
    Load flower classification dataset
    
    Parameters:
    -----------
    data_path : str
        Path to directory containing data files
        
    Returns:
    --------
    X : DataFrame
        Feature matrix (4065 x 512)
    y : array
        Labels (4065,)
    filenames : DataFrame
        Image filenames
    label_mapping : DataFrame
        Class name mapping
    class_names : array
        Array of class names
    """
    X_train = pd.read_csv(f'{data_path}flower_train_features.csv')
    y_train = pd.read_csv(f'{data_path}flower_train_labels.csv')['label'].values
    filenames = pd.read_csv(f'{data_path}flower_train_filenames.csv')
    label_mapping = pd.read_csv(f'{data_path}flower_label_mapping.csv')
    class_names = label_mapping['class_name'].values
    
    # Verify data shapes
    print(f"Loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    return X_train, y_train, filenames, label_mapping, class_names


def get_scaler(scaler_name='standard'):
    """
    Get scaler object by name
    
    Parameters:
    -----------
    scaler_name : str
        One of: 'standard', 'minmax', 'robust'
        
    Returns:
    --------
    scaler : sklearn transformer
    """
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    return scalers.get(scaler_name.lower(), StandardScaler())


def compare_normalizations(X, y, cv, classifier):
    """
    Compare different normalization methods
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Labels
    cv : cross-validation generator
    classifier : sklearn estimator
        Classifier to test with
        
    Returns:
    --------
    results : DataFrame
        Comparison results with accuracy scores
    """
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    
    scalers = {
        'None': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    results = []
    for scaler_name, scaler in scalers.items():
        if scaler is None:
            scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
        else:
            pipeline = Pipeline([
                ('scaler', scaler),
                ('classifier', classifier)
            ])
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        
        results.append({
            'Scaler': scaler_name,
            'Mean_Accuracy': scores.mean(),
            'Std': scores.std(),
            'Scores': scores
        })
    
    return pd.DataFrame(results)

