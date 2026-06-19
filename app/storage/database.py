from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "article_structuring.db"


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS structure_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                input_text TEXT NOT NULL,
                predicted_template TEXT NOT NULL,
                output_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def save_structure_request(title: str, input_text: str, predicted_template: str, output_json: dict) -> int:
    init_db()
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO structure_requests (
                title, input_text, predicted_template, output_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                title,
                input_text,
                predicted_template,
                json.dumps(output_json, ensure_ascii=False, indent=2),
                created_at,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
