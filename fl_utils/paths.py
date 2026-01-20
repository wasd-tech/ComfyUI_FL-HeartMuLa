"""
Path utilities for FL HeartMuLa.
Handles model directory resolution and package paths.
"""

import os
from pathlib import Path


def get_models_directory() -> Path:
    """
    Get the ComfyUI models/heartmula directory.
    Creates the directory if it doesn't exist.

    Returns:
        Path to models/heartmula directory
    """
    try:
        import folder_paths
        base = Path(folder_paths.models_dir)
    except Exception:
        # Fallback for non-ComfyUI environments
        base = Path(__file__).parent.parent.parent.parent.parent / "models"

    models_dir = base / "heartmula"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_heartlib_path() -> Path:
    """
    Get path to the bundled heartlib source.

    Returns:
        Path to heartlib directory within the node pack
    """
    return Path(__file__).parent.parent / "heartlib"


def get_package_root() -> Path:
    """
    Get the root directory of the FL-HeartMuLa package.

    Returns:
        Path to ComfyUI_FL-HeartMuLa directory
    """
    return Path(__file__).parent.parent


def ensure_heartlib_in_path():
    """
    Ensure the heartlib directory is in Python path for imports.
    """
    import sys
    heartlib_parent = str(get_package_root())
    if heartlib_parent not in sys.path:
        sys.path.insert(0, heartlib_parent)
