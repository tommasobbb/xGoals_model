from pathlib import Path
import yaml

# Load YAML config
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Extract paths
COMPETITIONS_PATH = Path(config["paths"]["competitions"])
MATCHES_ROOT      = Path(config["paths"]["matches_root"])
EVENTS_ROOT       = Path(config["paths"]["events_root"])
SHOTS_OUT_PATH    = Path(config["paths"]["shots_out"])
FEATURES_OUT_PATH = Path(config["paths"]["features_out"])
RESULTS_PATH = Path(config["paths"]["results"])
TRAINED_MODELS_PATH = Path(config["paths"]["trained_models"])