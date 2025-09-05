import os
from pathlib import Path

# Program root path
PROGRAM_ROOT = Path(__file__).resolve().parent.parent

# GLM database path
DB_PATH = PROGRAM_ROOT / 'data' / 'glm_data.db'

# Seconds between GLM data check
INGEST_REFRESH_SECONDS = int(os.environ.get('INGEST_POLL_SEC', '60'))

# Bucket name
BUCKET_NAME = os.environ.get('INGEST_BUCKET', 'noaa-goes19')

# Bucket prefix
PREFIX = os.environ.get('INGEST_PREFIX', 'GLM-L2-LCFA/')

# Max files to check
MAX_FILES = int(os.environ.get('INGEST_MAX_FILES', '10'))

# Check files in the last number of hours
INGEST_RECENT_HOURS = int(os.environ.get('INGEST_RECENT_HOURS', '6'))

# Max number of hours to store
RETENTION_HOURS = int(os.environ.get('RETENTION_HOURS', '6'))