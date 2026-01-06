
import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DreamerDataset
from src.visualizations import plot_vad_scatter, plot_vad_bars, plot_correlation_heatmap

def main():
    dataset = DreamerDataset('data/DREAMER.mat')
    try:
        dataset.load_data()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    all_valence = []
    all_arousal = []
    all_dominance = []

    print("Aggregating labels from all subjects...")
    for sub in range(1, dataset.subjects + 1):
        try:
            data = dataset.get_subject_data(sub)
            # data['valence'] is (18, 1) or (18,)
            all_valence.extend(data['valence'].flatten())
            all_arousal.extend(data['arousal'].flatten())
            all_dominance.extend(data['dominance'].flatten())
        except Exception as e:
            print(f"Error reading subject {sub}: {e}")

    # Convert to arrays
    valence = np.array(all_valence)
    arousal = np.array(all_arousal)
    dominance = np.array(all_dominance)

    print(f"Total samples: {len(valence)}")

    # Create plots
    output_dir = 'plots'
    print("Generating plots...")
    
    # 1. Scatter
    plot_vad_scatter(valence, arousal, output_dir)
    
    # 2. Distributions
    plot_vad_bars(valence, arousal, dominance, output_dir)
    
    # 3. Correlation
    df = pd.DataFrame({'Valence': valence, 'Arousal': arousal, 'Dominance': dominance})
    plot_correlation_heatmap(df, output_dir)

    print("Exploratory Analysis Complete.")

if __name__ == "__main__":
    main()
