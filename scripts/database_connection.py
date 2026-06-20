import sqlite3
import logging
from pathlib import Path
from typing import Optional, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "database" / "sleep_eeg.db"

def get_connection(db_path: Optional[Union[Path, str]] = None) -> sqlite3.Connection:
    path_to_use = Path(db_path) if db_path else DEFAULT_DB_PATH
    
    try:
        path_to_use.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path_to_use, timeout=10.0)
        conn.execute("PRAGMA foreign_keys = ON;")
        logger.debug(f"Connected to database successfully: {path_to_use}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to the database at {path_to_use}. Error: {e}")
        raise