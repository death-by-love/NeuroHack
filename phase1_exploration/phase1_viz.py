import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# --------------------------------------------------
# Helper: safely extract scalar from MATLAB values
# --------------------------------------------------
def to_scalar(x):
    if isinstance(x, np.ndarray):
        return float(x.squeeze())
    return float(x)

# --------------------------------------------------
# Load DREAMER.mat (SAFE & CORRECT)
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
mat_path = PROJECT_ROOT / "data" / "DREAMER.mat"

mat = sio.loadmat(mat_path)
dreamer = mat["DREAMER"][0, 0]          # MATLAB struct
data = dreamer["Data"]                  # Subject-wise cell array

no_subjects = int(dreamer["noOfSubjects"].squeeze())
no_videos = int(dreamer["noOfVideoSequences"].squeeze())

print(f"Subjects: {no_subjects}, Videos per subject: {no_videos}")

# --------------------------------------------------
# Extract Valence, Arousal, Dominance (FINAL LOGIC)
# --------------------------------------------------
records = []

for subj in range(no_subjects):
    # Each subject is a (1,1) cell
    subject_cell = data[0, subj]
    trials_array = subject_cell[0, 0]   # contains trials + possible metadata

    video_idx = 0

    for raw_trial in trials_array:
        trial = raw_trial

        # 🔑 unwrap all ndarray layers
        while isinstance(trial, np.ndarray):
            trial = trial.item()

        # keep ONLY real trial structs with emotion scores
        if not hasattr(trial, "dtype"):
            continue
        if trial.dtype.names is None:
            continue
        if "ScoreValence" not in trial.dtype.names:
            continue

        valence = to_scalar(trial["ScoreValence"])
        arousal = to_scalar(trial["ScoreArousal"])
        dominance = to_scalar(trial["ScoreDominance"])

        records.append([
            subj + 1,
            video_idx + 1,
            valence,
            arousal,
            dominance
        ])

        video_idx += 1

# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------
df = pd.DataFrame(
    records,
    columns=["Subject", "Video", "Valence", "Arousal", "Dominance"]
)

print("\nExtracted rows:", len(df))
print(df.head())

# --------------------------------------------------
# 1️⃣ Arousal–Valence Scatter Plot
# --------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(df["Valence"], df["Arousal"], alpha=0.6)
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.title("Arousal–Valence Emotional Space (DREAMER)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 2️⃣ Bar Chart: Mean VAD
# --------------------------------------------------
means = df[["Valence", "Arousal", "Dominance"]].mean()

plt.figure(figsize=(7, 5))
means.plot(kind="bar")
plt.ylabel("Mean Rating")
plt.title("Mean Valence, Arousal, Dominance")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3️⃣ Correlation Analysis
# --------------------------------------------------
corr = df[["Valence", "Arousal", "Dominance"]].corr()

print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(3), corr.columns)
plt.yticks(range(3), corr.columns)
plt.title("Correlation Between VAD Dimensions")
plt.tight_layout()
plt.show()
