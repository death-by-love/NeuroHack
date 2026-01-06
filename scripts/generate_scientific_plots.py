
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import welch

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DreamerDataset
from src.preprocessing import apply_bandpass_filter, apply_notch_filter, apply_car, apply_asr_lite
from src.features import compute_psd_features
from src.modeling import train_evaluate_loso
from sklearn.ensemble import RandomForestClassifier

# Set Style
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)
OUTPUT_DIR = 'plots/scientific_inferences'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_processed_data():
    X = np.load('output/X.npy')
    y = np.load('output/y.npy')
    groups = np.load('output/groups.npy')
    return X, y, groups

def plot_feature_stats(X, y):
    """Generates Graphs 9, 10, 12, 13 (Statistical Boxplots)"""
    print("Generating Feature Statistics Plots...")
    
    # Feature indices (based on extract_features.py structure)
    # 0-41: EEG (14ch * 3bands [Theta, Alpha, Beta])
    # Structure: Ch1[T,A,B], Ch2[T,A,B]... 
    # Actually extract_features logic: feats[ch, i] -> flattened. 
    # So: Ch1_T, Ch1_A, Ch1_B, Ch2_T...
    
    # We need to average Alpha across Frontal Channels (F3, F4, AF3, AF4)
    # Channels: 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
    # Indices: 0, 2, 11, 13 for AF3, F3, F4, AF4
    
    # Alpha is index 1 in [T, A, B]. 
    # Stride is 3. 
    # AF3_Alpha index = 0*3 + 1 = 1
    # F3_Alpha index = 2*3 + 1 = 7
    # F4_Alpha index = 11*3 + 1 = 34
    # AF4_Alpha index = 13*3 + 1 = 40
    
    alpha_indices = [1, 7, 34, 40]
    
    # Extract Alpha Power (Rows=Samples)
    alpha_power = np.mean(X[:, alpha_indices], axis=1)
    
    # Valence Labels (Median Split)
    valence_scores = y[:, 0]
    med_v = np.median(valence_scores)
    valence_labels = ['High Valence' if s >= med_v else 'Low Valence' for s in valence_scores]
    
    # Graph 9: Alpha Power by Valence
    plt.figure()
    sns.boxplot(x=valence_labels, y=alpha_power, palette="viridis")
    plt.title("Graph 9: Frontal Alpha Power by Valence (Inference 7)")
    plt.ylabel("Log Alpha Power")
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/graph09_alpha_valence.png')
    plt.close()

    # Graph 12: Heart Rate by Arousal
    # ECG Features are at the end. 
    # X shape is 88. 
    # End: [..., Asym, HR, SDNN, RMSSD]
    # HR is index -3
    hr_values = X[:, -3]
    
    arousal_scores = y[:, 1]
    med_a = np.median(arousal_scores)
    arousal_labels = ['High Arousal' if s >= med_a else 'Low Arousal' for s in arousal_scores]
    
    plt.figure()
    sns.boxplot(x=arousal_labels, y=hr_values, palette="rocket")
    plt.title("Graph 12: Heart Rate by Arousal (Inference 10)")
    plt.ylabel("Heart Rate (BPM)")
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/graph12_hr_arousal.png')
    plt.close()

def plot_preprocessing_impact():
    """Generates Graphs 4, 6, 8 (Signal Processing)"""
    print("Generating Signal Processing Plots...")
    
    # Load 1 Subject Trial
    dataset = DreamerDataset('data/DREAMER.mat')
    dataset.load_data()
    sub_data = dataset.get_subject_data(1)
    raw = sub_data['eeg_stimuli'][0].T # (14, samples)
    fs = dataset.sampling_rate
    
    # Filter
    filt = apply_notch_filter(apply_bandpass_filter(raw, fs), fs)
    
    # Graph 4: PSD Overlay
    f, Pxx_raw = welch(raw[0], fs, nperseg=fs*2)
    f, Pxx_filt = welch(filt[0], fs, nperseg=fs*2)
    
    plt.figure(figsize=(10, 5))
    plt.semilogy(f, Pxx_raw, label='Raw Signal', color='red', alpha=0.7)
    plt.semilogy(f, Pxx_filt, label='Filtered (1-45Hz, Notch 50Hz)', color='blue')
    plt.axvline(50, color='gray', linestyle='--')
    plt.annotate('50Hz Noise Removed', xy=(50, np.min(Pxx_filt)*10), xytext=(55, np.min(Pxx_filt)*100),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlim(0, 60)
    plt.title("Graph 4: PSD Before vs After Filtering (Inference 3)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/graph04_psd_overlay.png')
    plt.close()
    
    # Graph 6: ASR (Simulated Burst)
    # create artificial burst
    noisy_channel = filt[0].copy()
    burst_start = 200
    burst_end = 250
    noisy_channel[burst_start:burst_end] += 500 # Add big spike
    
    # Apply ASR-lite logic (simply clip here for viz as per previous implementation logic)
    # Recalculating ASR locally to show effect
    std_dev = np.std(noisy_channel)
    mean_val = np.mean(noisy_channel)
    clean_channel = np.clip(noisy_channel, mean_val - 20*std_dev, mean_val + 20*std_dev) # aggressive for viz
    
    t = np.arange(len(noisy_channel)) / fs
    
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(t, noisy_channel, color='orange')
    axes[0].set_title('Before ASR (Artifact Burst)')
    axes[0].axvspan(burst_start/fs, burst_end/fs, color='red', alpha=0.2)
    
    axes[1].plot(t, clean_channel, color='green')
    axes[1].set_title('After ASR (Burst Repaired)')
    axes[1].axvspan(burst_start/fs, burst_end/fs, color='green', alpha=0.1)
    
    plt.xlabel('Time (s)')
    plt.suptitle("Graph 6: ASR Artifact Removal (Inference 5)")
    plt.savefig(f'{OUTPUT_DIR}/graph06_asr_demo.png')
    plt.close()
    
    # Graph 8: PSD w/ Bands
    plt.figure()
    plt.plot(f, Pxx_filt, color='black')
    plt.fill_between(f, Pxx_filt, where=((f>=4)&(f<=8)), color='skyblue', alpha=0.5, label='Theta (4-8Hz)')
    plt.fill_between(f, Pxx_filt, where=((f>=8)&(f<=13)), color='gold', alpha=0.5, label='Alpha (8-13Hz)')
    plt.fill_between(f, Pxx_filt, where=((f>=13)&(f<=30)), color='salmon', alpha=0.5, label='Beta (13-30Hz)')
    plt.xlim(0, 40)
    plt.legend()
    plt.title("Graph 8: EEG Spectral Bands (Inference 3)")
    plt.savefig(f'{OUTPUT_DIR}/graph08_psd_bands.png')
    plt.close()

def plot_model_insights(X, y, groups):
    """Generates Graphs 14, 20, 21 (Model Performance & Features)"""
    print("Generating Model Insight Plots...")
    
    # Train RF for Feature Importance (Valence)
    valence_labels = (y[:, 0] >= np.median(y[:, 0])).astype(int)
    
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X, valence_labels)
    
    # Graph 20: Feature Importance
    importances = rf.feature_importances_
    # We have 88 features. Let's just plot top 10.
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(10, 8))
    plt.title("Graph 20: Top 15 Important Features (Valence)")
    plt.barh(range(15), importances[indices], align='center', color='teal')
    plt.yticks(range(15), [f"Feat #{i}" for i in indices]) # Generic names for now
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/graph20_feature_importance.png')
    plt.close()
    
    # Graph 21: Per-Subject Accuracy Boxplot (Simulated based on fold scores)
    # We need to run LOSO to get these scores.
    results = train_evaluate_loso(X, valence_labels, groups, model_type='xgb')
    # Use 'accuracy_folds' if we modify modeling.py to return it, OR just rerun loop here.
    # Current modeling.py returns means.
    # Let's trust the "Accuracy per fold" if we had it. 
    # I'll implement a quick loop here for the plot.
    
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import accuracy_score
    import xgboost as xgb
    
    logo = LeaveOneGroupOut()
    scores = []
    
    for train_idx, test_idx in logo.split(X, valence_labels, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = valence_labels[train_idx], valence_labels[test_idx]
        
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        clf.fit(X_train, y_train)
        scores.append(accuracy_score(y_test, clf.predict(X_test)))
        
    plt.figure()
    sns.boxplot(y=scores, color='lightblue')
    sns.stripplot(y=scores, color='black', alpha=0.5)
    plt.title("Graph 21: Per-Subject Accuracy Distribution (Inference 11)")
    plt.ylabel("Accuracy")
    plt.axhline(0.5, color='red', linestyle='--', label='Chance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/graph21_subject_variability.png')
    plt.close()
    
    # Graph 14: Bar Chart of scores
    plt.figure(figsize=(12, 5))
    plt.bar(range(1, len(scores)+1), scores, color='cornflowerblue')
    plt.axhline(np.mean(scores), color='red', linestyle='-', label=f'Mean: {np.mean(scores):.2f}')
    plt.xlabel("Subject ID")
    plt.ylabel("Accuracy")
    plt.title("Graph 14: LOSO Accuracy per Subject (Inference 11)")
    plt.legend()
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/graph14_loso_bars.png')
    plt.close()

def main():
    X, y, groups = load_processed_data()
    plot_feature_stats(X, y)
    plot_preprocessing_impact()
    plot_model_insights(X, y, groups)
    print(f"All Science Plots generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
