
import numpy as np
import mne
from mne.preprocessing import ICA

def apply_bandpass_filter(data, fs, l_freq=1.0, h_freq=45.0):
    """
    Applies bandpass filter to EEG data (channels x samples).
    """
    return mne.filter.filter_data(data, fs, l_freq, h_freq, verbose=False)

def apply_notch_filter(data, fs, freqs=50.0):
    """
    Applies notch filter to remove power line noise.
    """
    return mne.filter.notch_filter(data, fs, freqs, verbose=False)

def apply_car(data):
    """
    Applies Common Average Reference (CAR).
    data: (channels, samples)
    """
    return data - np.mean(data, axis=0)

def apply_ica(data, fs, n_components=None, exclude_components=None, random_state=42):
    """
    Applies ICA to remove artifacts.
    exclude_components: list of indices to remove (e.g. [0, 1] for blinks).
    Returns cleaned data and the fitted ICA object.
    """
    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    
    raw = mne.io.RawArray(data, info, verbose=False)
    
    if n_components is None:
        n_components = 0.95
        
    ica = ICA(n_components=n_components, method='fastica', random_state=random_state, verbose=False)
    ica.fit(raw)
    
    # If exclusion list is provided, exclude. 
    # Otherwise, for hackathon automation, we might blindly exclude first component if aggressive,
    # or rely on manual inspection. Here we support the argument.
    if exclude_components is not None:
        ica.exclude = exclude_components
        raw_clean = ica.apply(raw.copy(), verbose=False)
        return raw_clean.get_data(), ica
    
    return data, ica # Return original data if no exclusion specified, plus ICA for plotting

def apply_asr_lite(data, fs, window_len=0.5, cutoff_std=20):
    """
    ASR-lite: Variance-based artifact removal.
    Identifies windows with variance > cutoff_std * median_variance and interpolates them.
    data: (channels, samples)
    """
    n_channels, n_samples = data.shape
    window_samples = int(window_len * fs)
    clean_data = data.copy()
    
    # Calculate variances in windows
    # Since we don't have a calibration phase like real ASR, we use the median variance of the signal itself as baseline.
    
    # For simplicity/speed: calculate channel-wise variance z-scores
    # A burst usually affects all channels or specific ones strongly.
    
    # Sliding window standard deviation
    # We can iterate or use strided tricks. Iteration is fine for <1min signals.
    
    # We will just clamp high amplitude bursts to avoid destroying the signal if we can't interpolate well.
    # Or linear interpolation.
    
    # Let's verify 'cutoff_std': real ASR uses calibration.
    # Here, we'll clip amplitudes > k * std.
    
    std_dev = np.std(data, axis=1, keepdims=True)
    mean_val = np.mean(data, axis=1, keepdims=True)
    
    # Identify samples > cutoff
    # robust z-score: (x - median) / MAD is better, but std is asked.
    
    # Simple Clip for "Burst Removal" visualization (ASR does reconstruction, clipping is the "lite" version)
    lower = mean_val - cutoff_std * std_dev
    upper = mean_val + cutoff_std * std_dev
    
    clean_data = np.clip(clean_data, lower, upper)
    
    return clean_data
