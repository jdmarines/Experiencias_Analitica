import os
import argparse
import pandas as pd
import wandb
from sklearn.preprocessing import StandardScaler

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

execution_id = args.IdExecution or "testing-console"
print(f"IdExecution: {execution_id}")

def preprocess(X, normalize=True):
    """
    Optionally normalizes features using StandardScaler.
    """
    if normalize:
        X = StandardScaler().fit_transform(X)
    return X

def read_split(data_dir, split):
    """
    Reads a CSV split and returns features and target.
    """
    df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    return X, y

def preprocess_and_log(normalize=True):
    """
    Preprocesses the dataset and logs a new artifact to Weights & Biases.
    """
    with wandb.init(
        project="Diabetes",
        name=f"Preprocess Data ExecId-{execution_id}",
        job_type="preprocess-data"
    ) as run:

        # Load the latest raw diabetes data
        raw_data_artifact = run.use_artifact('diabetes-raw:latest')
        raw_data_path = raw_data_artifact.download(root="./data/artifacts/")

        processed_data = wandb.Artifact(
            "diabetes-preprocessed",
            type="dataset",
            description="Preprocessed Diabetes dataset (normalized)",
            metadata={"normalize": normalize}
        )

        for split in ["training", "validation", "test"]:
            X, y = read_split(raw_data_path, split)
            X_processed = preprocess(X, normalize=normalize)

            df = pd.DataFrame(X_processed, columns=[f"feature_{i}" for i in range(X_processed.shape[1])])
            df["target"] = y

            with processed_data.new_file(f"{split}.csv", mode="w") as file:
                df.to_csv(file, index=False)

        run.log_artifact(processed_data)

if __name__ == "__main__":
    preprocess_and_log(normalize=True)
