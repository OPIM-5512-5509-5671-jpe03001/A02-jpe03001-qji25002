import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

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
# Step 2: Calculate evaluation metrics
# ----------------------------------------------------
r2 = r2_score(y_train, y_train_pred)
mse = mean_squared_error(y_train, y_train_pred)
rmse = mse ** 0.5

print("\n--- Training Partition Evaluation ---")
print(f"R^2 Score: {r2:.4f}")
print(f"RMSE:      {rmse:.4f}")
print(f"MSE:       {mse:.4f}")

# ----------------------------------------------------
# Step 3: Plot Actual vs. Predicted for Training Set
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
# Step 4: Save the plot to figures/
# ----------------------------------------------------
output_path = Path("figures") / "train_actual_vs_pred.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output_path, dpi=300)
plt.close()
print(f"Saved actual vs. predicted plot to: {output_path}")
