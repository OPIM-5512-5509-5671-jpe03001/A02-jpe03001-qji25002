import matplotlib.pyplot as plt
from mlp_model import model
from train_test_split import X_test, y_test

# Generate predictions for the test set
y_test_pred = model.predict(X_test)

# Create actual vs. predicted plot
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
plt.savefig("figures/test_actual_vs_pred.png", dpi=300, bbox_inches="tight")
plt.close()

print("Test actual vs. predicted plot saved.")