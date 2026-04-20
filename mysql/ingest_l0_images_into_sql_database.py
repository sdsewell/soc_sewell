#!/usr/bin/env python3
"""
ingest_l0_images.py
-------------------
Scans /windcube/synthetic/raw for WindCube Level-0 .bin image files
and inserts their metadata into the windcube.l0_images MySQL table.

Expected filename format:  YYYY-MM-DDTHH-MM-SSZ_<type>.bin
  e.g.  2024-03-15T09-22-00Z_science.bin
        2024-03-15T09-22-00Z_calibration.bin
        2024-03-15T09-22-00Z_dark.bin

Usage:
    python ingest_l0_images.py

Requirements:
    pip install mysql-connector-python
"""

import os
import sys
import logging
from datetime import datetime
import mysql.connector
from mysql.connector import Error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "host":     "webdev.hao.ucar.edu",
    "user":     "windcuberw",
    "password": "",               # <-- fill in, or load from env / config file
    "database": "windcube",
    "port":     3306,
}

IMAGE_DIR   = "/windcube/synthetic/raw"
VALID_TYPES = {"science": "Science", "calibration": "Calibration", "dark": "Dark"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ingest_l0_images.log"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

def parse_filename(filename):
    """
    Parse a WindCube L0 filename into (datetime, image_type).

    Expected format:
        YYYY-MM-DDTHH-MM-SSZ_<type>.bin
        e.g. 2024-03-15T09-22-00Z_science.bin

    Returns (datetime object, type string) or raises ValueError.
    """
    name = os.path.splitext(filename)[0]          # strip .bin
    parts = name.split("_", 1)                    # split on first underscore only
    if len(parts) != 2:
        raise ValueError(f"Cannot split timestamp and type in: {filename}")

    ts_str, type_str = parts
    type_str = type_str.lower()

    if type_str not in VALID_TYPES:
        raise ValueError(
            f"Unknown image type '{type_str}' in {filename}. "
            f"Expected one of: {list(VALID_TYPES.keys())}"
        )

    # Strip trailing Z then parse  YYYY-MM-DDTHH-MM-SS
    ts_clean = ts_str.rstrip("Z")
    try:
        dt = datetime.strptime(ts_clean, "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        raise ValueError(f"Cannot parse timestamp '{ts_str}' in: {filename}")

    return dt, VALID_TYPES[type_str]

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def connect():
    """Open and return a database connection."""
    conn = mysql.connector.connect(**DB_CONFIG)
    log.info("Connected to %s@%s/%s", DB_CONFIG["user"],
             DB_CONFIG["host"], DB_CONFIG["database"])
    return conn


def insert_image(cursor, filename, timestamp, image_type):
    """
    Insert one image record. Skips silently if filename already exists
    (UNIQUE constraint) rather than raising an error — safe for re-runs.
    """
    sql = """
        INSERT IGNORE INTO l0_images (filename, timestamp, image_type)
        VALUES (%s, %s, %s)
    """
    cursor.execute(sql, (filename, timestamp, image_type))
    return cursor.rowcount   # 1 = inserted, 0 = already existed

# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

def ingest(image_dir):
    if not os.path.isdir(image_dir):
        log.error("Image directory not found: %s", image_dir)
        sys.exit(1)

    # Collect .bin files
    bin_files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(".bin")
    )
    if not bin_files:
        log.warning("No .bin files found in %s", image_dir)
        return

    log.info("Found %d .bin file(s) in %s", len(bin_files), image_dir)

    inserted = skipped = errors = 0

    try:
        conn   = connect()
        cursor = conn.cursor()

        for filename in bin_files:
            try:
                dt, img_type = parse_filename(filename)
            except ValueError as e:
                log.warning("SKIP  %s — %s", filename, e)
                errors += 1
                continue

            rows = insert_image(cursor, filename, dt, img_type)

            if rows == 1:
                log.info("INSERT  %s  |  %s  |  %s", filename, dt, img_type)
                inserted += 1
            else:
                log.info("EXISTS  %s — already in database, skipping", filename)
                skipped += 1

        conn.commit()
        log.info(
            "Done. Inserted: %d  |  Already existed: %d  |  Errors: %d",
            inserted, skipped, errors
        )

    except Error as e:
        log.error("Database error: %s", e)
        sys.exit(1)

    finally:
        if "cursor" in dir():
            cursor.close()
        if "conn" in dir() and conn.is_connected():
            conn.close()
            log.info("Database connection closed")


if __name__ == "__main__":
    ingest(IMAGE_DIR)
