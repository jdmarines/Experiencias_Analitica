import argparse
import pandas as pd
import wandb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

def load_data(train_size=0.8, val_size=0.1, random_state=42):
    """
    Loads and splits the Diabetes dataset into train, validation, and test sets.
    """
    data = load_diabetes()
    X_train, X_temp, y_train, y_temp = train_test_split(
        data.data, data.target, train_size=train_size, random_state=random_state
    )
    val_ratio = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, random_state=random_state
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_dataframe(X, y):
    """
    Creates a pandas DataFrame from features and target.
    """
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df

def load_and_log():
    """
    Loads data, formats it as DataFrames, and logs it to Weights & Biases.
    """
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Load Raw Data ExecId-{args.IdExecution}",
        job_type="load-data"
    ) as run:
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
        datasets = {
            "training": create_dataframe(X_train, y_train),
            "validation": create_dataframe(X_val, y_val),
            "test": create_dataframe(X_test, y_test)
        }

        artifact = wandb.Artifact(
            name="diabetes-raw",
            type="dataset",
            description="Diabetes dataset split into train/val/test",
            metadata={"source": "sklearn.datasets.load_diabetes",
                      "sizes": [len(df) for df in datasets.values()]}
        )

        for name, df in datasets.items():
            with artifact.new_file(f"{name}.csv", mode="w") as f:
                df.to_csv(f, index=False)

        run.log_artifact(artifact)

if __name__ == "__main__":
    load_and_log()
