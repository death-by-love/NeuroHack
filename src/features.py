
import numpy as np
from scipy.signal import welch

def compute_psd_features(data, fs, output_type='relative'):
    """
    Computes PSD for Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz).
    data: (channels, samples)
    returns: (channels, 3) where columns are [theta, alpha, beta]
    """
    n_channels, n_samples = data.shape
    features = np.zeros((n_channels, 3))
    
    # Welch's method parameters
    nperseg = min(fs * 2, n_samples) # 2-second windows
    
    # Frequencies of interest
    bands = [(4, 8), (8, 13), (13, 30)]
    
    for ch in range(n_channels):
        freqs, psd = welch(data[ch], fs, nperseg=nperseg)
        
        total_power = np.trapz(psd, freqs)
        
        for i, (fmin, fmax) in enumerate(bands):
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_power = np.trapz(psd[idx], freqs[idx])
            
            if output_type == 'relative':
                features[ch, i] = band_power / total_power if total_power > 0 else 0
            else:
                features[ch, i] = band_power
                
    return features

def compute_asymmetry(psd_features, channels):
    """
    Computes Frontal Alpha Asymmetry (log(F4) - log(F3)).
    channels: list of channel names
    psd_features: (channels, 3) where column 1 is Alpha
    """
    alpha_idx = 1
    try:
        f3_idx = channels.index('F3')
        f4_idx = channels.index('F4')
        
        f3_alpha = psd_features[f3_idx, alpha_idx]
        f4_alpha = psd_features[f4_idx, alpha_idx]
        
        # Avoid log(0)
        f3_alpha = f3_alpha if f3_alpha > 1e-10 else 1e-10
        f4_alpha = f4_alpha if f4_alpha > 1e-10 else 1e-10
        
        asymmetry = np.log(f4_alpha) - np.log(f3_alpha)
        return asymmetry
    except ValueError:
        return 0.0

def compute_hjorth_parameters(data):
    """
    Computes Hjorth parameters: Activity, Mobility, Complexity.
    data: (channels, samples)
    Returns: (channels, 3)
    """
    n_channels, n_samples = data.shape
    features = np.zeros((n_channels, 3))
    
    for ch in range(n_channels):
        x = data[ch]
        dx = np.diff(x)
        ddx = np.diff(dx)
        
        # Activity: Variance of time series
        activity = np.var(x)
        
        # Mobility: sqrt(var(dx) / var(x))
        mobility = np.sqrt(np.var(dx) / (activity + 1e-10))
        
        # Complexity: mobility(dx) / mobility(x)
        # mobility(dx) = sqrt(var(ddx) / var(dx))
        mobility_dx = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-10))
        complexity = mobility_dx / (mobility + 1e-10)
        
        features[ch, 0] = activity
        features[ch, 1] = mobility
        features[ch, 2] = complexity
        
    return features

def compute_ecg_features(ecg_data, fs):
    """
    Computes HRV features from ECG.
    ecg_data: (samples,) or (channels, samples)
    """
    # Simple R-peak detection using threshold or find_peaks
    from scipy.signal import find_peaks
    
    if ecg_data.ndim > 1:
        ecg_data = ecg_data[0] # Take first channel if multiple
        
    peaks, _ = find_peaks(ecg_data, distance=fs*0.6) # Assuming HR < 100bpm => >0.6s
    
    if len(peaks) < 2:
        return {'hr': 0, 'sdnn': 0, 'rmssd': 0}
        
    rr_intervals = np.diff(peaks) / fs * 1000 # in ms
    
    hr = 60000 / np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
    
    return {'hr': hr, 'sdnn': sdnn, 'rmssd': rmssd}
