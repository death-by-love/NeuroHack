
import os
import shutil

def organize_plots():
    DEST = 'plots/final_presentation_set'
    os.makedirs(DEST, exist_ok=True)
    
    # Mapping: (Source Path) -> (Destination Name)
    mapping = {
        # 1. Arousal-Valence Scatter
        'plots/vad_scatter.png': '01_Arousal_Valence_Scatter.png',
        
        # 2. VAD Correlation Heatmap
        'plots/vad_correlation.png': '02_VAD_Correlation_Heatmap.png',
        
        # 3. VAD Bar Charts
        'plots/missing_graphs/graph03_vad_bars.png': '03_VAD_Bar_Charts.png',
        
        # 4. PSD Before/After Filtering
        'plots/scientific_inferences/graph04_psd_overlay.png': '04_PSD_Filtering.png',
        
        # 5. ICA Component Topographies (Assuming existing)
        'plots/preprocessing/03_ica_components.png': '05_ICA_Components.png',
        
        # 6. ASR Before/After Window
        'plots/scientific_inferences/graph06_asr_demo.png': '06_ASR_Effect.png',
        
        # 7. CAR Effect
        'plots/missing_graphs/graph07_car_effect.png': '07_CAR_Effect.png',
        
        # 8. PSD with Band Highlights
        'plots/scientific_inferences/graph08_psd_bands.png': '08_PSD_Bands.png',
        
        # 9. Alpha Power by Valence
        'plots/scientific_inferences/graph09_alpha_valence.png': '09_Alpha_Valence.png',
        
        # 10. Beta Power by Arousal
        'plots/missing_graphs/graph10_beta_arousal.png': '10_Beta_Arousal.png',
        
        # 11. Baseline Correction Impact
        'plots/missing_graphs/graph11_baseline_impact.png': '11_Baseline_Correction.png',
        
        # 12. Heart Rate by Arousal
        'plots/scientific_inferences/graph12_hr_arousal.png': '12_Heart_Rate_Arousal.png',
        
        # 13. LOSO Validation Results (Bars)
        'plots/scientific_inferences/graph14_loso_bars.png': '13_LOSO_Results.png',
        
        # 14. Performance Comparison (VAD)
        'plots/missing_graphs/graph15_model_comparison.png': '14_Performance_Comparison.png',
        
        # 15. Confusion Matrices (3 files)
        'plots/results/cm_xgb_Valence.png': '15a_CM_Valence.png',
        'plots/results/cm_xgb_Arousal.png': '15b_CM_Arousal.png',
        'plots/results/cm_xgb_Dominance.png': '15c_CM_Dominance.png',
        
        # 16. Class Distribution
        'plots/missing_graphs/graph17_class_distribution.png': '16_Class_Distribution.png',
        
        # 17. Topography Maps (Valence + Arousal)
        'plots/topography/alpha_asymmetry.png': '17a_Topo_Valence.png',
        'plots/topography/arousal_beta.png': '17b_Topo_Arousal.png',
        
        # 18. Feature Importance
        'plots/scientific_inferences/graph20_feature_importance.png': '18_Feature_Importance.png',
        
        # 19. Per-Subject Variability
        'plots/scientific_inferences/graph21_subject_variability.png': '19_Subject_Variability.png',
        
        # 20. HRV Features (Missing, skipping or using HR if user meant that. I'll stick to what I have)
        
        # 21. Preprocessing SNR (Using Graph 4 again or ICA cleaning?)
        # Let's use ICA PSD clean comparison if available 'plots/preprocessing/04_ica_clean_psd.png'
        'plots/preprocessing/04_ica_clean_psd.png': '21_Preprocessing_SNR.png' 
    }
    
    print(f"Organizing files into {DEST}...")
    count = 0
    for src, dst in mapping.items():
        if os.path.exists(src):
            shutil.copy(src, os.path.join(DEST, dst))
            print(f"✅ Copied {dst}")
            count += 1
        else:
            print(f"⚠️ Missing: {src}")
            
    print(f"\nDone. {count} files organized.")

if __name__ == "__main__":
    organize_plots()
