
import scipy.io
import numpy as np
import os

class DreamerDataset:
    def __init__(self, mat_path):
        self.mat_path = mat_path
        self.data = None
        self.subjects = 23
        self.trials = 18
        self.sampling_rate = 128
        self.channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        
    def load_data(self):
        if not os.path.exists(self.mat_path):
            raise FileNotFoundError(f"File not found: {self.mat_path}")
            
        print(f"Loading {self.mat_path} (this might take a while)...")
        try:
            mat = scipy.io.loadmat(self.mat_path)
            self.data = mat['DREAMER']
            print("Dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def get_subject_data(self, subject_id):
        """
        Returns dictionary with EEG, ECG, and Labels for a specific subject (1-23)
        """
        if self.data is None:
            self.load_data()
            
        if subject_id < 1 or subject_id > self.subjects:
            raise ValueError(f"Subject ID must be between 1 and {self.subjects}")
        
        # Adjust for 0-indexing
        sub_idx = subject_id - 1
        
        # DREAMER structure handling (nested structs)
        # Assuming DREAMER -> Data -> {subject} -> EEG/ECG/etc
        # We need to navigate this carefully as scipy.io loads mats as nested numpy voids/arrays
        
        dreamer = self.data[0, 0] # Enter the struct
        # Fields usually: Data, C, ...
        data_struct = dreamer['Data']
        
        subject_data = data_struct[0, sub_idx]
        
        # Extract EEG
        eeg_struct = subject_data['EEG'][0, 0]
        eeg_baseline = eeg_struct['baseline'][0, 0]
        eeg_stimuli = eeg_struct['stimuli'][0, 0]

        # Extract ECG
        ecg_struct = subject_data['ECG'][0, 0]
        ecg_baseline = ecg_struct['baseline'][0, 0]
        ecg_stimuli = ecg_struct['stimuli'][0, 0]
        
        # Extract Labels (Valence, Arousal, Dominance)
        # usually in subject_data['ScoreValence'] etc.
        valence = subject_data['ScoreValence'][0, 0]
        arousal = subject_data['ScoreArousal'][0, 0]
        dominance = subject_data['ScoreDominance'][0, 0]
        
        return {
            'eeg_baseline': [eeg_baseline[i, 0] for i in range(self.trials)],
            'eeg_stimuli': [eeg_stimuli[i, 0] for i in range(self.trials)],
            'ecg_baseline': [ecg_baseline[i, 0] for i in range(self.trials)],
            'ecg_stimuli': [ecg_stimuli[i, 0] for i in range(self.trials)],
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance
        }

    def print_summary(self):
        if self.data is None:
            self.load_data()
        
        print(f"DREAMER Dataset Summary:")
        print(f"Subjects: {self.subjects}")
        print(f"Trials per subject: {self.trials}")
        print(f"Channels: {self.channels}")
        print(f"Sampling Rate: {self.sampling_rate} Hz")
        
        # Inspect first subject
        sub1 = self.get_subject_data(1)
        print("\nSubject 1 Check:")
        print(f"  EEG Baseline Trial 1 shape: {sub1['eeg_baseline'][0].shape}")
        print(f"  EEG Stimuli Trial 1 shape: {sub1['eeg_stimuli'][0].shape}")
        print(f"  Valence scores shape: {sub1['valence'].shape}")

if __name__ == "__main__":
    dataset = DreamerDataset('data/DREAMER.mat')
    dataset.print_summary()
