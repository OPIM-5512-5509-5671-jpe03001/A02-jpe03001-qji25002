from sklearn.neural_network import MLPRegressor
from train_test_split import X_train, y_train

# Create the MLP regression model
model = MLPRegressor(
    hidden_layer_sizes=(100,),
    early_stopping=True,
    random_state=42,
    max_iter=500
)

# Train the model
model.fit(X_train, y_train)

print("MLPRegressor training complete.")