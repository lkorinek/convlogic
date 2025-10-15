import os

import yaml


def load_config(config_name):
    config_name = config_name.replace("-", "_")
    config_path = f"configs/config_{config_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    with open(config_path) as f:
        return yaml.safe_load(f)
