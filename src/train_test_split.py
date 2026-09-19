import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure src directory is in sys.path for clean imports
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import df created in ds_pipeline
from ds_pipeline import df


def split_data(data, target_col="MedHouseVal", test_size=0.2, random_state=42):
    """
    Separates features and target variable, then performs train/test split.

    Parameters:
        data (pd.DataFrame): Dataset containing features and target column.
        target_col (str): Column name of target variable (default: 'MedHouseVal').
        test_size (float): Proportion for test split (default: 0.2).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Separate training features (X) from target variable (y)
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Perform train/test split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Execute train/test split
X_train, X_test, y_train, y_test = split_data(df)

# Quick check
print("\n--- Train / Test Split Summary ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

