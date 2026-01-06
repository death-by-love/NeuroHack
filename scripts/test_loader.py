
import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DreamerDataset

def test_loader():
    print("Testing patched Data Loader...")
    dataset = DreamerDataset('data/DREAMER.mat')
    dataset.load_data()
    
    # Test Subject 1
    try:
        data = dataset.get_subject_data(1)
        print("✅ Subject 1 loaded successfully.")
        print(f"  Valence shape: {data['valence'].shape} (Expected: (18, 1))")
        print(f"  EEG Stimuli Trial 0 shape: {data['eeg_stimuli'][0].shape}")
        
        if data['valence'].shape == (18, 1):
             print("✅ Dimensions look correct.")
        else:
             print("❌ Dimensions mismatch!")
             
    except Exception as e:
        print(f"❌ Failed to get subject 1: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loader()
