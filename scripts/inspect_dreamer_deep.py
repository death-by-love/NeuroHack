
import scipy.io
import numpy as np
import os

def print_structure(data, indent=0):
    indent_str = " " * indent
    if isinstance(data, dict):
        for key in data:
            if key.startswith('__'): continue
            value = data[key]
            print(f"{indent_str}{key}: {type(value)}")
            if isinstance(value, (dict, np.ndarray, np.void)):
                print_structure(value, indent + 2)
    elif isinstance(data, np.ndarray):
        print(f"{indent_str}Shape: {data.shape}, Dtype: {data.dtype}")
        # Only print details if it's small or we are specifically looking for the failure point
        if data.size < 10:
             print(f"{indent_str}Values: {data}")

def inspect_subject_1_deep(mat_path):
    print(f"Loading {mat_path}...")
    try:
        mat = scipy.io.loadmat(mat_path)
        dreamer = mat['DREAMER']
        print(f"DREAMER shape: {dreamer.shape}")
        
        # Data struct
        data_struct = dreamer[0,0]['Data']
        print(f"Data struct shape: {data_struct.shape}")
        
        # Subject 1
        sub1 = data_struct[0, 0] # Subject 1
        print(f"Subject 1 keys: {sub1.dtype.names}")
        
        # EEG
        eeg = sub1['EEG']
        print(f"Subject 1 EEG shape: {eeg.shape}")
        
        eeg_inner = eeg[0,0]
        print(f"Subject 1 EEG[0,0] keys: {eeg_inner.dtype.names}")
        
        baseline = eeg_inner['baseline']
        print(f"EEG baseline shape: {baseline.shape}")
        
        stimuli = eeg_inner['stimuli']
        print(f"EEG stimuli shape: {stimuli.shape}")
        
        # Labels
        val = sub1['ScoreValence']
        print(f"ScoreValence shape: {val.shape}")
        
    except Exception as e:
        print(f"Error inspecting: {e}")

if __name__ == "__main__":
    inspect_subject_1_deep('data/DREAMER.mat')
