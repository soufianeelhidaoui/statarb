from pathlib import Path
import yaml

def load_params(path: str | Path = "config/params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
