import pathlib


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

# Base directories for the raw PPMI data
ALL_DATA_ROOT = PROJECT_ROOT / "ALL DATA FILES REPOSITORY"
MODEL1_ROOT = PROJECT_ROOT / "MODEL 1 DATASET"


def resolve_all_data_path(rel_path: str) -> pathlib.Path:
    """Return absolute path inside ALL DATA FILES REPOSITORY."""
    return ALL_DATA_ROOT / rel_path


def resolve_model1_path(rel_path: str) -> pathlib.Path:
    """Return absolute path inside MODEL 1 DATASET."""
    return MODEL1_ROOT / rel_path


def ensure_exists(path: pathlib.Path) -> pathlib.Path:
    """Raise a clear error if path does not exist."""
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def ensure_data_roots_exist() -> None:
    """Quick sanity check that the expected data roots exist."""
    ensure_exists(ALL_DATA_ROOT)
    ensure_exists(MODEL1_ROOT)


