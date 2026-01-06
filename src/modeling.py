
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_evaluate_loso(X, y, groups, model_type='rf'):
    """
    Performs LOSO Cross-Validation.
    X: (n_samples, n_features)
    y: (n_samples,) binary labels
    groups: (n_samples,) subject IDs for LOSO
    model_type: 'rf' or 'xgb'
    """
    logo = LeaveOneGroupOut()
    accuracy = []
    f1 = []
    
    y_true_all = []
    y_pred_all = []
    
    print(f"Starting LOSO CV with {logo.get_n_splits(X, y, groups)} folds...")
    
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                       class_weight='balanced', random_state=42)
        elif model_type == 'xgb':
            # Count positives for scale_pos_weight
            pos = np.sum(y_train == 1)
            neg = len(y_train) - pos
            scale = neg/pos if pos > 0 else 1
            clf = XGBClassifier(n_estimators=100, learning_rate=0.1, 
                              scale_pos_weight=scale, random_state=42, n_jobs=-1)
            
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f = f1_score(y_test, y_pred, average='macro')
        
        accuracy.append(acc)
        f1.append(f)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        
    return {
        'accuracy_mean': np.mean(accuracy),
        'accuracy_std': np.std(accuracy),
        'f1_mean': np.mean(f1),
        'y_true': y_true_all,
        'y_pred': y_pred_all
    }

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
