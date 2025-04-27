#commit
import os
import argparse
import pandas as pd
import wandb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

execution_id = args.IdExecution or "testing-console"
print(f"IdExecution: {execution_id}")

# Read CSVs
def read_csv_data(data_dir, split):
    filepath = os.path.join(data_dir, f"{split}.csv")
    df = pd.read_csv(filepath)
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    return X, y

# Train model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    return mse, mae, rmse, r2

# Train and Evaluate
def train_and_evaluate(experiment_id='0'):
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Train-Eval LinearRegression ExecId-{execution_id} ExperimentId-{experiment_id}",
        job_type="train-eval"
    ) as run:
        
        data_artifact = run.use_artifact('diabetes-preprocessed:latest')
        data_dir = data_artifact.download()

        X_train, y_train = read_csv_data(data_dir, "training")
        X_val, y_val = read_csv_data(data_dir, "validation")
        X_test, y_test = read_csv_data(data_dir, "test")

        model = train_model(X_train, y_train)

        # Validation
        val_mse, val_mae, val_rmse, val_r2 = evaluate_model(model, X_val, y_val)
        print(f"Validation RMSE: {val_rmse:.4f} | R2: {val_r2:.4f}")

        # Test
        test_mse, test_mae, test_rmse, test_r2 = evaluate_model(model, X_test, y_test)
        print(f"Test RMSE: {test_rmse:.4f} | R2: {test_r2:.4f}")

        # Log metrics
        wandb.log({
            "validation/mse": val_mse,
            "validation/mae": val_mae,
            "validation/rmse": val_rmse,
            "validation/r2": val_r2,
            "test/mse": test_mse,
            "test/mae": test_mae,
            "test/rmse": test_rmse,
            "test/r2": test_r2
        })

# Entry point
if __name__ == "__main__":
    train_and_evaluate()
