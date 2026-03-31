"""Local filesystem paths (batch folders, model dirs, etc.)."""

import os


def create_path(path: str) -> None:
    """Create a directory tree if it does not exist."""
    os.makedirs(path, exist_ok=True)


def createPath(filepath: str) -> None:
    """Notebook-era name for :func:`create_path`."""
    create_path(filepath)
