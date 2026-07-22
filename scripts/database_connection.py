"""SQLite connection utilities for the EEG project."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional, Union

try:
    from .config import DATABASE_PATH
except ImportError:
    # Supports direct execution/import from inside the scripts directory.
    from config import DATABASE_PATH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = DATABASE_PATH


def get_connection(
    db_path: Optional[Union[Path, str]] = None,
) -> sqlite3.Connection:
    """Create a SQLite connection with foreign-key enforcement enabled."""

    path_to_use = (
        Path(db_path).resolve()
        if db_path is not None
        else DEFAULT_DB_PATH
    )

    try:
        path_to_use.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(
            str(path_to_use),
            timeout=10.0,
        )
        connection.execute("PRAGMA foreign_keys = ON;")

        logger.debug(
            "Connected to database successfully: %s",
            path_to_use,
        )

        return connection

    except sqlite3.Error:
        logger.exception(
            "Failed to connect to the database at %s",
            path_to_use,
        )
        raise
