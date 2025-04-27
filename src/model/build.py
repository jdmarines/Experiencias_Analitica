#commit
import os
import argparse
import pickle
import wandb
from sklearn.linear_model import LinearRegression

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

execution_id = args.IdExecution or "testing-console"
print(f"IdExecution: {execution_id}")

# Create model directory if it doesn't exist
os.makedirs("./model", exist_ok=True)

# Model parameters
input_shape = 10  
model_filename = "linear_regression_diabetes.pkl"

def build_model_and_log(config, model, model_name="linear_regression_diabetes", model_description="Linear Regression model for diabetes prediction"):
    """Build the model, save it locally, and log it as a W&B artifact."""
    with wandb.init(
        project="Diabetes",
        name=f"Initialize Model ExecId-{execution_id}",
        job_type="initialize-model",
        config=config
    ) as run:
        
        model_artifact = wandb.Artifact(
            name=model_name,
            type="model",
            description=model_description,
            metadata=dict(config)
        )

        # Save model locally
        model_path = f"./model/{model_filename}"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Add model file to artifact and log
        model_artifact.add_file(model_path)
        wandb.save(model_path)
        run.log_artifact(model_artifact)

# Model configuration
model_config = {"input_shape": input_shape}

# Initialize untrained model
model = LinearRegression()

# Build, save, and log model
build_model_and_log(model_config, model)
