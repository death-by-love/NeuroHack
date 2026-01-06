
import numpy as np
import os
import sys

def inspect_outputs():
    print("Inspecting Feature Matrix and Labels...")
    
    if not os.path.exists('output/X.npy'):
        print("❌ Error: output/X.npy not found.")
        return

    X = np.load('output/X.npy')
    y = np.load('output/y.npy')
    groups = np.load('output/groups.npy') if os.path.exists('output/groups.npy') else None

    print(f"✅ Loaded X: {X.shape}, y: {y.shape}")
    
    # 1. Check for NaNs/Infs
    n_nans = np.isnan(X).sum()
    n_infs = np.isinf(X).sum()
    if n_nans > 0 or n_infs > 0:
        print(f"⚠️ WARNING: Found {n_nans} NaNs and {n_infs} Infs in Feature Matrix!")
    else:
        print("✅ No NaNs or Infs found in features.")

    # 2. Check Value Ranges
    print(f"Feature Statistics:")
    print(f"  Mean: {np.mean(X):.4f}")
    print(f"  Std:  {np.std(X):.4f}")
    print(f"  Min:  {np.min(X):.4f}")
    print(f"  Max:  {np.max(X):.4f}")

    # 3. Check Labels (Class Balance)
    # y is (samples, 3) for Valence, Arousal, Dominance (Scores 1-5 usually)
    print("\nLabel Statistics (Original Scores):")
    targets = ['Valence', 'Arousal', 'Dominance']
    for i, target in enumerate(targets):
        scores = y[:, i]
        print(f"  {target}: Mean={np.mean(scores):.2f}, Median={np.median(scores):.2f}, Min={np.min(scores)}, Max={np.max(scores)}")
        
        # Simulate Binarization (Median Split)
        threshold = np.median(scores)
        binary = (scores >= threshold).astype(int)
        balance = np.mean(binary)
        print(f"    -> Binary Balance (>= {threshold:.1f}): {balance*100:.1f}% Positive Class")

    if groups is not None:
        print(f"\nGroups (Subjects): {len(np.unique(groups))} unique subjects found.")
    
    print("\nConclusion: Data looks ready for modeling." if n_nans == 0 else "\nConclusion: Data needs cleaning (NaNs present).")

if __name__ == "__main__":
    inspect_outputs()
