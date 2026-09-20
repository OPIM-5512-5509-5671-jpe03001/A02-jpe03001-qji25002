# [INACTIVE] Preserved for project history/reference.
# The complete, active machine learning workflow is now unified in src/ds_pipeline.py.

import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure src directory is in sys.path for clean imports
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import df created in ds_pipeline
from ds_pipeline import df

# Step 1: Separate features (X) and target variable (y)
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Step 2: Train / test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Quick check of shapes
print("\n--- Train / Test Split Summary ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")
