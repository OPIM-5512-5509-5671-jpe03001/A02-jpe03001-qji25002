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

# Run the unified end-to-end pipeline
python .\src\ds_pipeline.py
```

### Pipeline Steps Executed in `src/ds_pipeline.py`:
1. **Data Loading & EDA**: Loads California Housing dataset and saves median house value boxplot.
2. **Train / Test Split**: Partitions data into 80% train and 20% test splits (`random_state=42`).
3. **Model Training**: Fits an `MLPRegressor` with `early_stopping=True`, `hidden_layer_sizes=(100,)`, and `max_iter=500`.
4. **Training Evaluation & Plotting**: Computes $R^2$, RMSE, and MSE on train split; saves original and revised 1:1 square plots.
5. **Test Evaluation & Plotting**: Computes $R^2$, RMSE, and MSE on test split; saves original and revised 1:1 square plots.

## Project Deliverables & Outputs (`figures/`)
* `figures/med_house_value_boxplot.png`: Exploratory distribution boxplot.
* `figures/train_actual_vs_pred.png`: Training partition actual vs. predicted plot.
* `figures/test_actual_vs_pred.png`: Test partition actual vs. predicted plot.
* `figures/train_actual_vs_pred_revised.png`: Training plot with 1:1 matching square axes and corner-to-corner reference line.
* `figures/test_actual_vs_pred_revised.png`: Test plot with 1:1 matching square axes and corner-to-corner reference line.

> **Note on Modular Scripts**: The initial modular scripts developed during the ping-pong PR iterations (`train_test_split_inactive.py`, `mlp_model_inactive.py`, `train_preds_plots_inactive.py`, and `test_preds_plot_inactive.py`) have been unified into `src/ds_pipeline.py` and are preserved in `src/` as inactive references for full project history.

