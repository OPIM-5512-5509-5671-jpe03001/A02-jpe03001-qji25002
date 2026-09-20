# [INACTIVE] Preserved for project history/reference.
# The complete, active machine learning workflow is now unified in src/ds_pipeline.py.

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure src directory is in sys.path for clean imports
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import trained model and training partition data from mlp_model
from mlp_model import model, X_train, y_train

# ----------------------------------------------------
# Step 1: Make predictions on the training partition
# ----------------------------------------------------
y_train_pred = model.predict(X_train)

# ----------------------------------------------------
# Step 2: Plot Actual vs. Predicted for Training Set
# ----------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, alpha=0.5, color="blue")
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2)
plt.title("Actual vs. Predicted - Training Set")
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.grid(True)
plt.tight_layout()

# ----------------------------------------------------
# Step 3: Save the plot to figures/
# ----------------------------------------------------
output_path = Path("figures") / "train_actual_vs_pred.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=300)
plt.close()
print(f"Saved actual vs. predicted plot to: {output_path}")

# ----------------------------------------------------
# Step 4: Revised Uniform Training Plot (1:1 Matching Square Axes)
# ----------------------------------------------------
# Shared limits and ticks so x and y change 1-for-1 together
COMMON_LIM = (-2.0, 8.5)
COMMON_TICKS = [-2, 0, 2, 4, 6, 8]

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_train, y_train_pred, alpha=0.4, color="tab:blue")
ax.plot([COMMON_LIM[0], COMMON_LIM[1]], [COMMON_LIM[0], COMMON_LIM[1]], "r--", lw=2, label="Perfect Prediction (y = x)")
ax.set_xlim(COMMON_LIM)
ax.set_ylim(COMMON_LIM)
ax.set_xticks(COMMON_TICKS)
ax.set_yticks(COMMON_TICKS)
ax.set_box_aspect(1)
ax.set_title("Actual vs. Predicted - Training Set")
ax.set_xlabel("Actual Median House Value")
ax.set_ylabel("Predicted Median House Value")
ax.grid(True)
plt.tight_layout()

revised_output_path = Path("figures") / "train_actual_vs_pred_revised.png"
plt.savefig(revised_output_path, dpi=300)
plt.close()
print(f"Saved revised actual vs. predicted plot to: {revised_output_path}")

