
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DreamerDataset
from src.preprocessing import apply_bandpass_filter, apply_notch_filter, apply_car, apply_ica
from scipy.signal import welch

def plot_psd(data, fs, title, save_path):
    f, Pxx = welch(data, fs, nperseg=fs*2)
    plt.figure()
    plt.semilogy(f, Pxx.T)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V**2/Hz)')
    plt.savefig(save_path)
    plt.close()

def main():
    # 1. Load Data (Subset or Mock for testing speed if loader fails, but let's try real)
    # To speed up, we might try to load just one part in data_loader if possible, but it's not.
    # We assume data is loaded in memory if we run this or we load it.
    
    dataset = DreamerDataset('data/DREAMER.mat')
    try:
        dataset.load_data()
    except Exception as e:
        print(f"Skipping preprocessing verification due to load error: {e}")
        return

    # Get one trial from Subject 1
    sub1 = dataset.get_subject_data(1)
    raw_eeg = sub1['eeg_stimuli'][0] # (samples, 14)
    raw_eeg = raw_eeg.T # (channels, samples)
    fs = dataset.sampling_rate

    output_dir = 'plots/preprocessing'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Inspect Raw PSD
    plot_psd(raw_eeg, fs, 'Raw PSD', f'{output_dir}/01_raw_psd.png')

    # 2. Filter (Bandpass + Notch)
    filtered = apply_bandpass_filter(raw_eeg, fs, 1.0, 45.0)
    filtered = apply_notch_filter(filtered, fs, 50.0)
    plot_psd(filtered, fs, 'Filtered PSD (1-45Hz, Notch 50Hz)', f'{output_dir}/02_filtered_psd.png')

    # 3. CAR
    car_data = apply_car(filtered)
    
    # 4. ICA (plot components + apply simple exclusion to show effect)
    ica_data, ica = apply_ica(car_data, fs, n_components=14, exclude_components=[0])
    # Generate ICA component plots
    fig = ica.plot_components(show=False)
    # MNE plot_components returns list of figures or figure
    if isinstance(fig, list):
        for i, f in enumerate(fig):
            f.savefig(f'{output_dir}/03_ica_components_{i}.png')
            plt.close(f)
    else:
        fig.savefig(f'{output_dir}/03_ica_components.png')
        plt.close(fig)
        
    # 5. PSD after ICA cleaning
    plot_psd(ica_data, fs, 'PSD after ICA (exclude [0])', f'{output_dir}/04_ica_clean_psd.png')
    
    print("Preprocessing verification plots generated (raw, filtered, ICA components, ICA-clean PSD).")

if __name__ == "__main__":
    main()
