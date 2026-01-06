
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DreamerDataset
from src.preprocessing import apply_bandpass_filter, apply_notch_filter, apply_car, apply_ica, apply_asr_lite
from src.features import compute_psd_features, compute_asymmetry, compute_ecg_features, compute_hjorth_parameters

def main():
    dataset = DreamerDataset('data/DREAMER.mat')
    try:
        dataset.load_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    fs = dataset.sampling_rate
    channels = dataset.channels
    
    features_list = []
    labels_list = []
    groups_list = []
    
    print("Extracting features (this will take time)...")
    
    # Process subjects
    for sub in range(1, dataset.subjects + 1):
        try:
            print(f"Processing Subject {sub}...")
            data = dataset.get_subject_data(sub)
            
            # For each trial (18 videos)
            for trial in range(dataset.trials):
                # 1. Baseline Correction (Ratio)
                
                # --- Preprocessing ---
                # Baseline
                raw_base = data['eeg_baseline'][trial].T # (ch, samples)
                raw_base = apply_notch_filter(apply_bandpass_filter(raw_base, fs), fs)
                raw_base = apply_car(raw_base)
                raw_base = apply_asr_lite(raw_base, fs)
                # ICA on baseline is usually overkill/unstable due to short duration, skipping or applying same weights?
                # For hackathon, just cleaning is enough.
                
                # Trial
                raw_trial = data['eeg_stimuli'][trial].T
                
                # Filter
                raw_trial = apply_notch_filter(apply_bandpass_filter(raw_trial, fs), fs)
                # ICA (Fit on trial, reject top 2 components blindly for automation or skipped if dangerous)
                # Let's do a safe cleaning: exclude 0 (often blink).
                # Changed to n_components=0.99 to avoid rank deficiency warning (since CAR removed 1 DoF)
                raw_trial, _ = apply_ica(raw_trial, fs, n_components=0.99, exclude_components=[0])
                
                # CAR
                raw_trial = apply_car(raw_trial)
                # ASR Lite
                raw_trial = apply_asr_lite(raw_trial, fs)
                
                # --- Feature Extraction ---
                # PSD Features
                feats_base = compute_psd_features(raw_base, fs, output_type='absolute')
                feats_trial = compute_psd_features(raw_trial, fs, output_type='absolute')
                
                # Relative Change
                feat_change = (feats_trial - feats_base) / (feats_base + 1e-10)
                vector_eeg = feat_change.flatten() # 14x3 = 42
                
                # Asymmetry
                asym = compute_asymmetry(compute_psd_features(raw_trial, fs, output_type='relative'), channels)
                
                # Hjorth (New) on Trial Data
                hjorth = compute_hjorth_parameters(raw_trial) # 14x3 = 42
                vector_hjorth = hjorth.flatten()
                
                # ECG Features
                ecg_trial = data['ecg_stimuli'][trial].T
                ecg_feats = compute_ecg_features(ecg_trial, fs)
                
                # Combine (42 + 42 + 1 + 3 = 88 features)
                feature_vector = np.concatenate([
                    vector_eeg, 
                    vector_hjorth, 
                    [asym], 
                    [ecg_feats['hr'], ecg_feats['sdnn'], ecg_feats['rmssd']]
                ])
                
                features_list.append(feature_vector)
                labels_list.append([data['valence'][trial, 0], data['arousal'][trial, 0], data['dominance'][trial, 0]])
                groups_list.append(sub) # Save subject ID
                
        except Exception as e:
            print(f"Error processing subject {sub}: {e}")
            continue

    # Save
    X = np.array(features_list)
    y = np.array(labels_list)
    groups = np.array(groups_list)
    
    print(f"Feature extraction complete. Shape: {X.shape}")
    
    os.makedirs('output', exist_ok=True)
    np.save('output/X.npy', X)
    np.save('output/y.npy', y)
    np.save('output/groups.npy', groups) # New
    print("Saved features, labels, and groups to output/")

if __name__ == "__main__":
    main()
