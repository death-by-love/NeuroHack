
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling import train_evaluate_loso, plot_confusion_matrix

def main():
    # Load features
    if not os.path.exists('output/X.npy'):
        print("Features not found. Please run scripts/extract_features.py")
        return
        
    print("Loading features...")
    X = np.load('output/X.npy')
    y_vad = np.load('output/y.npy')
    
    # Load Groups if available, else reconstructed (fallback)
    if os.path.exists('output/groups.npy'):
        groups = np.load('output/groups.npy')
        print(f"Loaded groups for {len(np.unique(groups))} subjects.")
    else:
        print("Groups file not found, creating dummy groups (NOT RECOMMENDED for LOSO).")
        groups = np.zeros(len(X)) 
    
    # Binarize Labels
    output_dir = 'plots/results'
    os.makedirs(output_dir, exist_ok=True)
    
    targets = ['Valence', 'Arousal', 'Dominance']
    
    for i, target in enumerate(targets):
        print(f"\n--- Training for {target} ---")
        y_score = y_vad[:, i]
        threshold = np.median(y_score)
        y_bin = (y_score >= threshold).astype(int)
        
        # Train RF
        results_rf = train_evaluate_loso(X, y_bin, groups, model_type='rf')
        print(f"RF Results: Acc={results_rf['accuracy_mean']:.3f}, F1={results_rf['f1_mean']:.3f}")
        
        plot_confusion_matrix(results_rf['y_true'], results_rf['y_pred'], 
                            f'{target} RF Confusion Matrix', 
                            f'{output_dir}/cm_rf_{target}.png')
        
        # Train XGBoost
        results_xgb = train_evaluate_loso(X, y_bin, groups, model_type='xgb')
        print(f"XGB Results: Acc={results_xgb['accuracy_mean']:.3f}, F1={results_xgb['f1_mean']:.3f}")
        
        plot_confusion_matrix(results_xgb['y_true'], results_xgb['y_pred'], 
                            f'{target} XGB Confusion Matrix', 
                            f'{output_dir}/cm_xgb_{target}.png')

if __name__ == "__main__":
    main()
