
import sys
import os
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import apply_bandpass_filter, apply_notch_filter, apply_car, apply_ica, apply_asr_lite
from src.features import compute_psd_features, compute_asymmetry, compute_ecg_features, compute_hjorth_parameters
from src.modeling import train_evaluate_loso

def test_pipeline():
    print("Testing Preprocessing & Feature Extraction Pipeline...")
    fs = 128
    n_channels = 14
    n_samples = 128 * 4 # 4 seconds
    
    # Mock Data
    eeg_data = np.random.randn(n_channels, n_samples)
    ecg_data = np.random.randn(1, n_samples)
    
    # 1. Preprocessing Integration
    try:
        filt = apply_bandpass_filter(eeg_data, fs)
        notch = apply_notch_filter(filt, fs)
        car = apply_car(notch)
        asr = apply_asr_lite(car, fs)
        clean_ica, _ = apply_ica(asr, fs, n_components=5, exclude_components=[0])
        print("✅ Preprocessing pipeline: SUCCESS")
    except Exception as e:
        print(f"❌ Preprocessing pipeline failed: {e}")
        return

    # 2. Feature Integration
    try:
        psd = compute_psd_features(clean_ica, fs)
        asym = compute_asymmetry(psd, ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'])
        hjorth = compute_hjorth_parameters(clean_ica)
        ecg = compute_ecg_features(ecg_data, fs)
        print("✅ Feature extraction pipeline: SUCCESS")
    except Exception as e:
        print(f"❌ Feature pipeline failed: {e}")
        return

    # 3. Modeling Integration
    try:
        # Mock Feature Matrix
        X = np.random.randn(20, 10) # 20 samples, 10 features
        y = np.random.randint(0, 2, 20)
        groups = np.array([1]*10 + [2]*10) # 2 subjects
        
        train_evaluate_loso(X, y, groups, model_type='rf')
        print("✅ Modeling pipeline: SUCCESS")
    except Exception as e:
        print(f"❌ Modeling pipeline failed: {e}")
        return

    print("\nAll integration tests PASSED. Code is jelled.")

if __name__ == "__main__":
    test_pipeline()
