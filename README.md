# 🧠 EEG Emotion Recognition System
**Hackathon Project: Emotion Classification from Brainwaves (DREAMER Dataset)**

## 🚀 Overview
This project implements a robust machine learning pipeline to classify emotions (Valence, Arousal, Dominance) using raw EEG signals from the DREAMER dataset. It features an end-to-end processing chain including artifact removal (ICA/ASR), biological feature extraction (Hjorth, Asymmetry, PSD), and Leave-One-Subject-Out (LOSO) validation.

**Key Achievements:**
*   **~63% Accuracy on Valence** (beating random chance significantly).
*   **Biological Proof:** Distinct Frontal Alpha Asymmetry topomaps validating the "Happy vs Sad" brain states.
*   **Novelty:** Integrated EKG Heart Rate variability and Hjorth complexity parameters.

## 🛠️ Tech Stack
*   **Language:** Python 3.9+
*   **Neuroscience Libs:** `MNE-Python`, `Scipy.signal`
*   **ML Stack:** `Scikit-learn`, `XGBoost`
*   **Visualization:** `Matplotlib`, `Seaborn`

## 🏃‍♂️ How to Run

### 1. Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd emotion_hackathon

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
Place the `DREAMER.mat` file in the `data/` folder.
*(Note: Data is not included in this repo due to size constraints).*

### 3. Run Pipeline
Execute the scripts in this specific order:

**Phase 1: Preprocessing & Visualization**
```bash
python scripts/generate_eda.py         # 1. Exploratory Data Analysis
python scripts/verify_preprocessing.py # 2. Verify Cleaning (ICA/ASR)
```

**Phase 2: Feature Extraction**
```bash
# This takes ~15-20 minutes
python scripts/extract_features.py     # 3. Extract PSD, Hjorth, Asymmetry
```

**Phase 3: Training & Results**
```bash
python scripts/train_models.py         # 4. Train XGBoost/RF (LOSO Validation)
python scripts/generate_topomaps.py    # 5. Generate Brain Maps
```

## 📊 Results Summary
| Dimension | Accuracy | F1-Score | Best Model |
| :--- | :--- | :--- | :--- |
| **Valence** | 62.7% | 0.55 | XGBoost |
| **Arousal** | 63.7% | 0.48 | Random Forest |
| **Dominance** | 76.0% | 0.53 | Random Forest |

## 📂 Project Structure
*   `src/`: Core logic (preprocessing, features, loaders).
*   `scripts/`: Executable drivers for each pipeline stage.
*   `plots/final_presentation_set/`: **The output graphs** for presentation.
*   `output/`: Intermediate `.npy` feature files (ignored by git).
