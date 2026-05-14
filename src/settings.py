from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


CFG = load_config()

IMG_SIZE = int(CFG["image"]["size"])
CHANNELS = int(CFG["image"]["channels"])
CLASS_NAMES = list(CFG["classes"])
NUM_CLASSES = len(CLASS_NAMES)
CLASS_DISPLAY = dict(CFG["class_display"])

RAW_DIR = project_path(CFG["paths"]["raw_dataset"])
FIELD_DIR = project_path(CFG["paths"]["field_dataset"])
PREPARED_DIR = project_path(CFG["paths"]["prepared_root"])
MODELS_DIR = project_path(CFG["paths"]["models"])
REPORTS_DIR = project_path(CFG["paths"]["reports"])

for d in (MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
