
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_vad_scatter(valence, arousal, output_dir='plots'):
    """
    Plots Valence vs Arousal 2D scatter plot with quadrants.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=valence, y=arousal, alpha=0.6, edgecolor=None)
    
    # Add quadrants
    plt.axhline(y=3, color='k', linestyle='--', alpha=0.5) # Assuming 1-5 scale (DREAMER is 1-5)
    plt.axvline(x=3, color='k', linestyle='--', alpha=0.5)
    
    plt.text(4.5, 4.5, 'HVHA', fontsize=12, fontweight='bold', ha='center')
    plt.text(1.5, 4.5, 'LVHA', fontsize=12, fontweight='bold', ha='center')
    plt.text(1.5, 1.5, 'LVLA', fontsize=12, fontweight='bold', ha='center')
    plt.text(4.5, 1.5, 'HVLA', fontsize=12, fontweight='bold', ha='center')
    
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title('2D Arousal-Valence Scatter Plot')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'vad_scatter.png'))
    plt.close()
    print(f"Saved vad_scatter.png to {output_dir}")

def plot_vad_bars(valence, arousal, dominance, output_dir='plots'):
    """
    Plots bar charts for V, A, D distributions.
    """
    plt.figure(figsize=(15, 5))
    
    dims = {'Valence': valence, 'Arousal': arousal, 'Dominance': dominance}
    
    for i, (name, data) in enumerate(dims.items(), 1):
        plt.subplot(1, 3, i)
        sns.histplot(data, bins=20, kde=True)
        plt.title(f'{name} Distribution')
        plt.xlabel('Score')
        plt.ylabel('Count')
        
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'vad_distribution.png'))
    plt.close()
    print(f"Saved vad_distribution.png to {output_dir}")

def plot_correlation_heatmap(df, output_dir='plots'):
    """
    Plots correlation heatmap for VAD dataframe.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('VAD Correlation Matrix')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'vad_correlation.png'))
    plt.close()
    print(f"Saved vad_correlation.png to {output_dir}")
