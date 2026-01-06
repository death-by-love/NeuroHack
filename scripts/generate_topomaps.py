
import sys
import os
import numpy as np
import mne
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DreamerDataset
from src.preprocessing import apply_bandpass_filter, apply_car

def main():
    dataset = DreamerDataset('data/DREAMER.mat')
    try:
        dataset.load_data()
    except:
        print("Could not load data. Ensure DREAMER.mat is in data/")
        return

    fs = dataset.sampling_rate
    info = mne.create_info(dataset.channels, fs, 'eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # Accumulate Alpha power for High/Low Valence
    # We need to compute power per channel, average across subs
    
    alpha_power_hv = [] # High Valence
    alpha_power_lv = [] # Low Valence
    
    # Analyze a subset to save time (e.g. first 5 subjects)
    for sub in range(1, min(6, dataset.subjects + 1)):
        data = dataset.get_subject_data(sub)
        val = data['valence']
        median_val = np.median(val)
        
        for i in range(dataset.trials):
            # Get Trial Data
            raw = data['eeg_stimuli'][i].T
            # Preprocess
            raw = apply_car(apply_bandpass_filter(raw, fs, 8, 13)) # Alpha band only
            
            # Compute Power (RMS^2 or Variance)
            power = np.var(raw, axis=1) # (14,)
            
            if val[i] >= median_val:
                alpha_power_hv.append(power)
            else:
                alpha_power_lv.append(power)
                
    # Average
    avg_hv = np.mean(alpha_power_hv, axis=0)
    avg_lv = np.mean(alpha_power_lv, axis=0)
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    im, _ = mne.viz.plot_topomap(avg_hv, info, axes=ax[0], show=False, cmap='viridis')
    ax[0].set_title('High Valence (Alpha)')
    plt.colorbar(im, ax=ax[0])
    
    im, _ = mne.viz.plot_topomap(avg_lv, info, axes=ax[1], show=False, cmap='viridis')
    ax[1].set_title('Low Valence (Alpha)')
    plt.colorbar(im, ax=ax[1])
    
    output_dir = 'plots/topography'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/alpha_asymmetry.png')
    print(f"Saved topography map to {output_dir}/alpha_asymmetry.png")

if __name__ == "__main__":
    main()
