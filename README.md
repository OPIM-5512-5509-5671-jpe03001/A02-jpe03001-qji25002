# A02-jpe03001-qji25002: Ping Pong Assignment

Collaborative machine learning project demonstrating GitHub branch/PR collaboration while training an MLP regression model.

## Collaborators
- James Eastwood (`jpe03001`)
- Eldhose Kochakkan Varghesekutty (`qji25002`)

## Project Workflow
- **GitHub Workflow**: Multi-branch collaboration (`branch -> pull request -> review -> merge -> delete branch`) completing at least 5 ping-pong interactions.
- **Task**: Train an `MLPRegressor` on the California Housing dataset using early stopping and custom hyperparameters, saving actual vs. predicted evaluation plots to `figures/`.

## Running the Pipeline
```bash
# Setup virtual environment
python -m venv venv
venv\Scripts\activate  # Windows (or: source venv/bin/activate on Mac/Linux)

# Install dependencies
pip install -r requirements.txt

# Run the training script
python .\src\ds_pipeline.py
```
