# [INACTIVE] Preserved for project history/reference.
# The complete, active machine learning workflow is now unified in src/ds_pipeline.py.

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure src directory is in sys.path for clean imports
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from mlp_model import model
from train_test_split import X_test, y_test

# Generate predictions for the test set
y_test_pred = model.predict(X_test)

# ----------------------------------------------------
# Original Test Plot
# ----------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.4)

# Add reference line for perfect predictions
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--")

plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs. Predicted - Test Set")
plt.grid(True)

# Save the required test plot
output_path = Path("figures") / "test_actual_vs_pred.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("Test actual vs. predicted plot saved.")

# ----------------------------------------------------
# Revised Uniform Test Plot (1:1 Matching Square Axes)
# ----------------------------------------------------
# Shared limits and ticks so x and y change 1-for-1 together
COMMON_LIM = (-2.0, 8.5)
COMMON_TICKS = [-2, 0, 2, 4, 6, 8]

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_test_pred, alpha=0.4, color="tab:blue")
ax.plot([COMMON_LIM[0], COMMON_LIM[1]], [COMMON_LIM[0], COMMON_LIM[1]], "r--", lw=2, label="Perfect Prediction (y = x)")
ax.set_xlim(COMMON_LIM)
ax.set_ylim(COMMON_LIM)
ax.set_xticks(COMMON_TICKS)
ax.set_yticks(COMMON_TICKS)
ax.set_box_aspect(1)
ax.set_title("Actual vs. Predicted - Test Set")
ax.set_xlabel("Actual Median House Value")
ax.set_ylabel("Predicted Median House Value")
ax.grid(True)
plt.tight_layout()

revised_output_path = Path("figures") / "test_actual_vs_pred_revised.png"
plt.savefig(revised_output_path, dpi=300)
plt.close()
print(f"Saved revised actual vs. predicted plot to: {revised_output_path}")
