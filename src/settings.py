from pathlib import Path

# Program root path
PROGRAM_ROOT = Path(__file__).resolve().parent.parent

# GLM database path
DB_PATH = PROGRAM_ROOT / 'data' / 'glm_data.db'

# Seconds between GLM data check
INGEST_REFRESH_SECONDS = 60