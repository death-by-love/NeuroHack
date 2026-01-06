
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import welch
import mne

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DreamerDataset
from src.preprocessing import apply_bandpass_filter, apply_notch_filter, apply_car
from src.modeling import train_evaluate_loso

OUTPUT_DIR = 'plots/missing_graphs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set(style="whitegrid", context="paper", font_scale=1.2)

def load_processed():
    X = np.load('output/X.npy')
    y = np.load('output/y.npy')
    return X, y

def plot_vad_bars(y):
    """Graph 3: VAD Bar Charts"""
    print("Generating Graph 3: VAD Bar Charts...")
    df = pd.DataFrame(y, columns=['Valence', 'Arousal', 'Dominance'])
    melted = df.melt(var_name='Dimension', value_name='Score')
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Dimension', y='Score', data=melted, capsize=.1, palette='muted', errorbar=('ci', 95))
    plt.title("Graph 3: Mean VAD Scores (with 95% CI)")
    plt.ylim(1, 5)
    plt.savefig(f'{OUTPUT_DIR}/graph03_vad_bars.png')
    plt.close()

def plot_car_effect():
    """Graph 7: CAR Effect Demonstration"""
    print("Generating Graph 7: CAR Effect...")
    dataset = DreamerDataset('data/DREAMER.mat')
    dataset.load_data()
    sub_data = dataset.get_subject_data(1)
    raw = sub_data['eeg_stimuli'][0].T # (14, samples)
    fs = dataset.sampling_rate
    
    # Filter first
    filt = apply_notch_filter(apply_bandpass_filter(raw, fs), fs)
    car = apply_car(filt)
    
    # Plot one channel (e.g., F3)
    t = np.arange(fs*2) / fs # 2 seconds
    ch_idx = 2 # F3
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, filt[ch_idx, :fs*2], color='red', label='Before CAR')
    plt.title("Before CAR (Common Noise Present)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    
    plt.subplot(1, 2, 2)
    plt.plot(t, car[ch_idx, :fs*2], color='blue', label='After CAR')
    plt.title("After CAR (Re-referenced)")
    plt.xlabel("Time (s)")
    
    plt.suptitle("Graph 7: Common Average Reference Effect")
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/graph07_car_effect.png')
    plt.close()

def plot_beta_arousal(X, y):
    """Graph 10: Beta Power by Arousal"""
    print("Generating Graph 10: Beta Power by Arousal...")
    # Beta indices: 2, 5, 8... (Index 2 in [T,A,B])
    # Average across all channels for robustness (or Frontal/Central)
    beta_indices = [i for i in range(2, 42, 3)]
    beta_power = np.mean(X[:, beta_indices], axis=1)
    
    arousal_labels = ['High Arousal' if s >= np.median(y[:, 1]) else 'Low Arousal' for s in y[:, 1]]
    
    plt.figure(figsize=(6, 6))
    sns.boxplot(x=arousal_labels, y=np.log(beta_power + 1e-5), palette="coolwarm")
    plt.title("Graph 10: Beta Power by Arousal")
    plt.ylabel("Log Beta Power")
    plt.savefig(f'{OUTPUT_DIR}/graph10_beta_arousal.png')
    plt.close()

def plot_baseline_impact():
    """Graph 11: Baseline Correction Impact"""
    print("Generating Graph 11: Baseline Correction Impact...")
    # We don't have raw baseline vs trial power saved in X.npy (it's ratio).
    # But we can simulate the CONCEPT by showing variance of raw baseline vs ratio.
    # Load 5 subjects
    dataset = DreamerDataset('data/DREAMER.mat')
    dataset.load_data()
    
    raw_powers = []
    ratio_powers = []
    
    for sub in range(1, 6):
        data = dataset.get_subject_data(sub)
        # Take trial 0
        base = data['eeg_baseline'][0].T
        trial = data['eeg_stimuli'][0].T
        fs = 128
        
        # Power
        f, pb = welch(base[0], fs)
        f, pt = welch(trial[0], fs)
        
        raw_powers.append(np.mean(pt)) 
        ratio_powers.append((np.mean(pt) - np.mean(pb))/np.mean(pb))
        
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(5), raw_powers, color='gray')
    plt.title("Raw Power (High Subject Variance)")
    plt.xlabel("Subject")
    plt.ylabel("Power")
    
    plt.subplot(1, 2, 2)
    plt.bar(range(5), ratio_powers, color='purple')
    plt.title("Relative Change (Normalized)")
    plt.xlabel("Subject")
    
    plt.suptitle("Graph 11: Baseline Correction Impact")
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/graph11_baseline_impact.png')
    plt.close()

def plot_model_comparison():
    """Graph 15: Performance Comparison (V vs A vs D)"""
    print("Generating Graph 15: Model Performance Comparison...")
    # Hardcoded from results (since re-running takes time and user knows them)
    # V: Acc 62.7, F1 0.55
    # A: Acc 63.7, F1 0.48
    # D: Acc 76.0, F1 0.53
    
    data = {
        'Dimension': ['Valence', 'Valence', 'Arousal', 'Arousal', 'Dominance', 'Dominance'],
        'Metric': ['Accuracy', 'F1-Score', 'Accuracy', 'F1-Score', 'Accuracy', 'F1-Score'],
        'Value': [0.627, 0.551, 0.637, 0.480, 0.760, 0.527]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Dimension', y='Value', hue='Metric', data=df, palette='viridis')
    plt.title("Graph 15: XGBoost Performance Comparison")
    plt.ylim(0, 1.0)
    plt.axhline(0.5, color='red', linestyle='--', label='Chance')
    plt.legend(loc='lower right')
    plt.savefig(f'{OUTPUT_DIR}/graph15_model_comparison.png')
    plt.close()

def plot_class_distribution(y):
    """Graph 17: Class Distribution (Binary)"""
    print("Generating Graph 17: Binary Class Distribution...")
    targets = ['Valence', 'Arousal', 'Dominance']
    counts = []
    
    for i, t in enumerate(targets):
        med = np.median(y[:, i])
        high = np.sum(y[:, i] >= med)
        low = np.sum(y[:, i] < med)
        counts.append({'Dimension': t, 'Class': 'High', 'Count': high})
        counts.append({'Dimension': t, 'Class': 'Low', 'Count': low})
        
    df = pd.DataFrame(counts)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Dimension', y='Count', hue='Class', data=df, palette='coolwarm')
    plt.title("Graph 17: Class Distribution (High vs Low)")
    plt.savefig(f'{OUTPUT_DIR}/graph17_class_distribution.png')
    plt.close()

def plot_arousal_topomaps(y):
    """Graph 19: Arousal Topography Maps (Beta)"""
    print("Generating Graph 19: Arousal Topomaps...")
    dataset = DreamerDataset('data/DREAMER.mat')
    dataset.load_data()
    
    # Info for plotting
    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    
    # We need beta power for high/low arousal trials
    # We'll use X.npy beta features if aligned, or quick re-extract from subset
    # For speed, let's process subject 1-5 only? No, must match labels.
    # Actually, we can just use the X.npy beta columns!
    X, _ = load_processed()
    # Beta indices: 2, 5, 8 ... (Start 2, stride 3, count 14)
    beta_indices = np.arange(2, 42, 3) 
    beta_data = X[:, beta_indices] # (403, 14)
    
    # Arousal Labels
    arousal_scores = y[:, 1]
    med = np.median(arousal_scores)
    high_idx = arousal_scores >= med
    low_idx = arousal_scores < med
    
    avg_beta_high = np.mean(beta_data[high_idx], axis=0)
    avg_beta_low = np.mean(beta_data[low_idx], axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    vm = np.max([np.abs(avg_beta_high), np.abs(avg_beta_low)])
    
    mne.viz.plot_topomap(avg_beta_high, info, axes=axes[0], show=False, cmap='RdBu_r', vlim=(0, vm))
    axes[0].set_title('High Arousal (Beta)')
    
    im, _ = mne.viz.plot_topomap(avg_beta_low, info, axes=axes[1], show=False, cmap='RdBu_r', vlim=(0, vm))
    axes[1].set_title('Low Arousal (Beta)')
    
    plt.colorbar(im, ax=axes)
    plt.savefig(f'plots/topography/arousal_beta.png') # Saving to topography folder
    plt.close()

def main():
    X, y = load_processed()
    plot_vad_bars(y)
    plot_car_effect()
    plot_beta_arousal(X, y)
    plot_baseline_impact()
    plot_model_comparison()
    plot_class_distribution(y)
    plot_arousal_topomaps(y)
    print(f"Missing Graphs generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
