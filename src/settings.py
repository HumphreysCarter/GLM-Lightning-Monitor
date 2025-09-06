import os
from pathlib import Path


def parse_bounds(env_var='DATA_BOUNDS', default=None):
    raw = os.environ.get(env_var)
    if not raw:
        if default is not None:
            return tuple(map(float, default))
        raise ValueError(f'{env_var} is not set and no default provided')

    parts = [p.strip() for p in raw.split(',') if p.strip() != '']
    if len(parts) != 4:
        raise ValueError(f'{env_var} must have 4 comma-separated numbers: min_lon,min_lat,max_lon,max_lat')

    try:
        bounds = tuple(float(p) for p in parts)  # (min_lon, min_lat, max_lon, max_lat)
    except ValueError as e:
        raise ValueError(f'{env_var} contains a non-numeric value') from e

    # Optional sanity checks (longitude/latitude ranges)
    min_lon, min_lat, max_lon, max_lat = bounds
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise ValueError('longitudes must be in [-180, 180]')
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValueError('latitudes must be in [-90, 90]')
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError('bounds must satisfy min < max for lon and lat')

    return bounds


# Program root path
PROGRAM_ROOT = Path(__file__).resolve().parent.parent

LOCAL_DATA_PATH = PROGRAM_ROOT / 'local_data'

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

# Get data bounds
DATA_BOUNDS = parse_bounds(default=(-131.5, 18.8, -62.8, 52.2))
