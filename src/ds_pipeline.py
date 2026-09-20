"""
End-to-End Data Science Pipeline: California Housing MLP Regression

This consolidated pipeline executes the complete machine learning workflow:
  1. Data Loading & Exploratory Data Analysis (Boxplot)
  2. Train / Test Split (80% train, 20% test)
  3. Model Training (MLPRegressor with early stopping)
  4. Training Evaluation & Predictions Plotting (Original + 1:1 Square Revised)
  5. Test Evaluation & Predictions Plotting (Original + 1:1 Square Revised)
"""

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Ensure figures output directory exists
figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)

# ====================================================================
# Step 1: Read the Data & Exploratory Data Analysis
# ====================================================================
print("=" * 60)
print("Step 1: Loading California Housing Dataset...")
print("=" * 60)

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"Data shape: {df.shape}")
print(df.head())

# Save boxplot of Median House Value
plt.figure(figsize=(6, 4))
df["MedHouseVal"].plot.box()
plt.title("Boxplot of Median House Value")
plt.ylabel("Median House Value")
plt.tight_layout()
boxplot_path = figures_dir / "med_house_value_boxplot.png"
plt.savefig(boxplot_path)
plt.close()
print(f"Saved boxplot to: {boxplot_path}")

# ====================================================================
# Step 2: Train / Test Split (80% Train, 20% Test)
# ====================================================================
print("\n" + "=" * 60)
print("Step 2: Train / Test Split...")
print("=" * 60)

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

# ====================================================================
# Step 3: Train MLPRegressor Model
# ====================================================================
print("\n" + "=" * 60)
print("Step 3: Training MLPRegressor with Early Stopping...")
print("=" * 60)

model = MLPRegressor(
    hidden_layer_sizes=(100,),
    early_stopping=True,
    random_state=42,
    max_iter=500,
)

model.fit(X_train, y_train)
print("MLPRegressor training complete.")

# ====================================================================
# Step 4: Training Evaluation & Predictions Plots
# ====================================================================
print("\n" + "=" * 60)
print("Step 4: Evaluating on Training Partition...")
print("=" * 60)

y_train_pred = model.predict(X_train)

r2_train = r2_score(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = mse_train ** 0.5

print(f"R^2 Score: {r2_train:.4f}")
print(f"RMSE:      {rmse_train:.4f}")
print(f"MSE:       {mse_train:.4f}")

# 4a: Original Training Actual vs. Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, alpha=0.5, color="blue")
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2)
plt.title("Actual vs. Predicted - Training Set")
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.grid(True)
plt.tight_layout()

train_plot_path = figures_dir / "train_actual_vs_pred.png"
plt.savefig(train_plot_path, dpi=300)
plt.close()
print(f"Saved actual vs. predicted plot to: {train_plot_path}")

# 4b: Revised Uniform Training Plot (1:1 Matching Square Axes)
COMMON_LIM = (-2.0, 8.5)
COMMON_TICKS = [-2, 0, 2, 4, 6, 8]

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_train, y_train_pred, alpha=0.4, color="tab:blue")
ax.plot(
    [COMMON_LIM[0], COMMON_LIM[1]],
    [COMMON_LIM[0], COMMON_LIM[1]],
    "r--",
    lw=2,
    label="Perfect Prediction (y = x)",
)
ax.set_xlim(COMMON_LIM)
ax.set_ylim(COMMON_LIM)
ax.set_xticks(COMMON_TICKS)
ax.set_yticks(COMMON_TICKS)
ax.set_box_aspect(1)
ax.set_title("Actual vs. Predicted - Training Set")
ax.set_xlabel("Actual Median House Value")
ax.set_ylabel("Predicted Median House Value")
ax.grid(True)

# Metrics annotation in the corner away from datapoints
train_metrics_text = f"R² = {r2_train:.4f}\nRMSE = {rmse_train:.4f}\nMSE = {mse_train:.4f}"
bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray")
ax.text(
    0.05,
    0.95,
    train_metrics_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=bbox_props,
)

plt.tight_layout()

revised_train_path = figures_dir / "train_actual_vs_pred_revised.png"
plt.savefig(revised_train_path, dpi=300)
plt.close()
print(f"Saved revised actual vs. predicted plot to: {revised_train_path}")

# ====================================================================
# Step 5: Test Evaluation & Predictions Plots
# ====================================================================
print("\n" + "=" * 60)
print("Step 5: Evaluating on Test Partition...")
print("=" * 60)

y_test_pred = model.predict(X_test)

r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = mse_test ** 0.5

print(f"R^2 Score: {r2_test:.4f}")
print(f"RMSE:      {rmse_test:.4f}")
print(f"MSE:       {mse_test:.4f}")

# 5a: Original Test Actual vs. Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.4)

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--")

plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs. Predicted - Test Set")
plt.grid(True)

test_plot_path = figures_dir / "test_actual_vs_pred.png"
plt.savefig(test_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved test actual vs. predicted plot to: {test_plot_path}")

# 5b: Revised Uniform Test Plot (1:1 Matching Square Axes)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_test_pred, alpha=0.4, color="tab:blue")
ax.plot(
    [COMMON_LIM[0], COMMON_LIM[1]],
    [COMMON_LIM[0], COMMON_LIM[1]],
    "r--",
    lw=2,
    label="Perfect Prediction (y = x)",
)
ax.set_xlim(COMMON_LIM)
ax.set_ylim(COMMON_LIM)
ax.set_xticks(COMMON_TICKS)
ax.set_yticks(COMMON_TICKS)
ax.set_box_aspect(1)
ax.set_title("Actual vs. Predicted - Test Set")
ax.set_xlabel("Actual Median House Value")
ax.set_ylabel("Predicted Median House Value")
ax.grid(True)

# Metrics annotation in the corner away from datapoints
test_metrics_text = f"R² = {r2_test:.4f}\nRMSE = {rmse_test:.4f}\nMSE = {mse_test:.4f}"
ax.text(
    0.05,
    0.95,
    test_metrics_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=bbox_props,
)

plt.tight_layout()

revised_test_path = figures_dir / "test_actual_vs_pred_revised.png"
plt.savefig(revised_test_path, dpi=300)
plt.close()
print(f"Saved revised test actual vs. predicted plot to: {revised_test_path}")

print("\n" + "=" * 60)
print("Pipeline complete! All figures successfully generated.")
print("=" * 60)
