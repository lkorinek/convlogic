import argparse
import os
import subprocess

import wandb
import yaml


def train_wrapper():
    with open("configs/sweep_config.yaml") as f:
        sweep_config = yaml.safe_load(f)

    parameters = sweep_config.get("parameters", {})

    cmd = ["python3", "-m", "train"]
    for key, val_dict in parameters.items():
        if isinstance(val_dict, dict) and "value" in val_dict:
            val = val_dict["value"]
            cmd.append(f"{key}={val}")

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", type=str, default=None, help="Optional existing sweep ID to run")
    parser.add_argument(
        "--project-name", type=str, default="my-logic-network-sweep", help="Optional project name for the sweep"
    )
    args = parser.parse_args()

    # Log in to W&B using the environment variable WANDB_API_KEY
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        print("No WANDB_API_KEY found. Please provide it via Docker ARG/ENV.")
    else:
        wandb.login(key=wandb_api_key)
        print("Logged in to Weights & Biases.")

    sweep_id = args.sweep_id
    project_name = args.project_name

    if sweep_id is None:
        with open("configs/sweep_config.yaml") as f:
            sweep_config = yaml.safe_load(f)

        # Create the sweep
        sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
        print(f"Created new sweep with ID: {sweep_id}")
    else:
        print(f"Using provided sweep ID: {sweep_id}")

    # Start the sweep agent
    wandb.agent(sweep_id, project=project_name, function=train_wrapper, count=100)


if __name__ == "__main__":
    main()
