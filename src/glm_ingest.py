import os
import time
import boto3
import heapq
import logging
import sqlite3
import tempfile
import netCDF4 as nc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from pathlib import Path

from .settings import DB_PATH, INGEST_REFRESH_SECONDS, BUCKET_NAME, PREFIX, MAX_FILES, INGEST_RECENT_HOURS, \
    RETENTION_HOURS, DATA_BOUNDS, LOCAL_DATA_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for local data directory in container
if LOCAL_DATA_PATH.exists() and LOCAL_DATA_PATH.is_dir():
    logger.info(f"Local data mode enabled: {LOCAL_DATA_PATH}")
else:
    LOCAL_DATA_PATH = None
    logger.info("Local data mode disabled - using AWS S3")


def _recent_glm_prefixes(root_prefix: str, recent_hours: int = 24) -> list[str]:
    now = datetime.now(timezone.utc)
    hours = [now - timedelta(hours=h) for h in range(recent_hours)]
    hours.sort(reverse=True)  # newest first
    prefixes = []
    root = (root_prefix or 'GLM-L2-LCFA').rstrip('/')
    for dt in hours:
        yyyy = dt.strftime('%Y')
        ddd = dt.strftime('%j')  # day-of-year
        hh = dt.strftime('%H')
        prefixes.append(f'{root}/{yyyy}/{ddd}/{hh}/')
    return list(dict.fromkeys(prefixes))


def _within_bounds(lon: float, lat: float, bounds: tuple[float, float, float, float]) -> bool:
    """Check if point is within geographic bounds."""
    min_lon, min_lat, max_lon, max_lat = bounds
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)


def _filter_data_by_bounds(data: Dict[str, Any], bounds: tuple[float, float, float, float]) -> Dict[str, Any]:
    """Filter extracted data by geographic bounds."""
    if not bounds:
        return data

    min_lon, min_lat, max_lon, max_lat = bounds
    filtered_data = {}

    logger.info(f"Applying geographic filter: lon=[{min_lon:.2f}, {max_lon:.2f}], lat=[{min_lat:.2f}, {max_lat:.2f}]")

    # Filter events data
    if 'events' in data and len(data['events']['event_id']) > 0:
        events = data['events']
        lons = events['event_lon']
        lats = events['event_lat']

        # Create boolean mask for points within bounds
        mask = ((lons >= min_lon) & (lons <= max_lon) &
                (lats >= min_lat) & (lats <= max_lat))

        if np.any(mask):
            filtered_events = {}
            for key, values in events.items():
                filtered_events[key] = values[mask]
            filtered_data['events'] = filtered_events
            logger.info(f"Events: {len(events['event_id'])} → {len(filtered_events['event_id'])} (within bounds)")
        else:
            logger.info(f"Events: {len(events['event_id'])} → 0 (none within bounds)")

    # Filter groups data
    if 'groups' in data and len(data['groups']['group_id']) > 0:
        groups = data['groups']
        lons = groups['group_lon']
        lats = groups['group_lat']

        mask = ((lons >= min_lon) & (lons <= max_lon) &
                (lats >= min_lat) & (lats <= max_lat))

        if np.any(mask):
            filtered_groups = {}
            for key, values in groups.items():
                filtered_groups[key] = values[mask]
            filtered_data['groups'] = filtered_groups
            logger.info(f"Groups: {len(groups['group_id'])} → {len(filtered_groups['group_id'])} (within bounds)")
        else:
            logger.info(f"Groups: {len(groups['group_id'])} → 0 (none within bounds)")

    # Filter flashes data
    if 'flashes' in data and len(data['flashes']['flash_id']) > 0:
        flashes = data['flashes']
        lons = flashes['flash_lon']
        lats = flashes['flash_lat']

        mask = ((lons >= min_lon) & (lons <= max_lon) &
                (lats >= min_lat) & (lats <= max_lat))

        if np.any(mask):
            filtered_flashes = {}
            for key, values in flashes.items():
                filtered_flashes[key] = values[mask]
            filtered_data['flashes'] = filtered_flashes
            logger.info(f"Flashes: {len(flashes['flash_id'])} → {len(filtered_flashes['flash_id'])} (within bounds)")
        else:
            logger.info(f"Flashes: {len(flashes['flash_id'])} → 0 (none within bounds)")

    return filtered_data


def init_database():
    '''Initialize SQLite database with GLM data tables.'''
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create events table (GLM lightning events)
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS glm_events
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       file_name
                       TEXT
                       NOT
                       NULL,
                       event_id
                       REAL,
                       event_time_offset
                       REAL,
                       event_lat
                       REAL,
                       event_lon
                       REAL,
                       event_energy
                       REAL,
                       parent_group_id
                       REAL,
                       created_at
                       TIMESTAMP
                       DEFAULT
                       CURRENT_TIMESTAMP
                   )
                   ''')

    # Create groups table (GLM lightning groups)
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS glm_groups
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       file_name
                       TEXT
                       NOT
                       NULL,
                       group_id
                       REAL,
                       group_time_offset
                       REAL,
                       group_lat
                       REAL,
                       group_lon
                       REAL,
                       group_area
                       REAL,
                       group_energy
                       REAL,
                       group_quality_flag
                       INTEGER,
                       parent_flash_id
                       REAL,
                       created_at
                       TIMESTAMP
                       DEFAULT
                       CURRENT_TIMESTAMP
                   )
                   ''')

    # Create flashes table (GLM lightning flashes)
    # Includes both first/last event offsets (as seen in your files) and keeps flash_time_offset for compatibility.
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS glm_flashes
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       file_name
                       TEXT
                       NOT
                       NULL,
                       flash_id
                       REAL,
                       flash_time_offset_of_first_event
                       REAL,
                       flash_time_offset_of_last_event
                       REAL,
                       flash_time_offset
                       REAL,
                       flash_lat
                       REAL,
                       flash_lon
                       REAL,
                       flash_area
                       REAL,
                       flash_energy
                       REAL,
                       flash_quality_flag
                       INTEGER,
                       created_at
                       TIMESTAMP
                       DEFAULT
                       CURRENT_TIMESTAMP
                   )
                   ''')

    # —— Migration guard: ensure glm_flashes has the expected columns even if the table already existed
    cursor.execute('PRAGMA table_info(glm_flashes)')
    existing_cols = {row[1] for row in cursor.fetchall()}
    alters = []
    if 'flash_time_offset_of_first_event' not in existing_cols:
        alters.append('ALTER TABLE glm_flashes ADD COLUMN flash_time_offset_of_first_event REAL')
    if 'flash_time_offset_of_last_event' not in existing_cols:
        alters.append('ALTER TABLE glm_flashes ADD COLUMN flash_time_offset_of_last_event REAL')
    if 'flash_time_offset' not in existing_cols:
        alters.append('ALTER TABLE glm_flashes ADD COLUMN flash_time_offset REAL')
    for stmt in alters:
        cursor.execute(stmt)

    # Create file processing log
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS processing_log
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       file_name
                       TEXT
                       UNIQUE
                       NOT
                       NULL,
                       file_size
                       INTEGER,
                       processing_status
                       TEXT,
                       processing_time
                       TIMESTAMP
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       error_message
                       TEXT,
                       records_filtered
                       INTEGER
                       DEFAULT
                       0,
                       records_inserted
                       INTEGER
                       DEFAULT
                       0
                   )
                   ''')

    # Add new columns to processing_log if they don't exist
    cursor.execute('PRAGMA table_info(processing_log)')
    existing_cols = {row[1] for row in cursor.fetchall()}
    if 'records_filtered' not in existing_cols:
        cursor.execute('ALTER TABLE processing_log ADD COLUMN records_filtered INTEGER DEFAULT 0')
    if 'records_inserted' not in existing_cols:
        cursor.execute('ALTER TABLE processing_log ADD COLUMN records_inserted INTEGER DEFAULT 0')

    # Create indexes for better query performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_time ON glm_events(event_time_offset)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_location ON glm_events(event_lat, event_lon)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_groups_time ON glm_groups(group_time_offset)')

    # Time indexes for flashes: index the 'first event' offset (primary), and also the legacy single offset if present
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_flashes_time_first ON glm_flashes(flash_time_offset_of_first_event)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_flashes_time_legacy ON glm_flashes(flash_time_offset)')

    # Indexes for deletes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_created ON glm_events(created_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_groups_created ON glm_groups(created_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_flashes_created ON glm_flashes(created_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_processing_log_time ON processing_log(processing_time)')

    conn.commit()
    conn.close()
    logger.info(f'Database initialized: {DB_PATH}')


def is_file_processed(file_name: str) -> bool:
    """Check if file has already been processed successfully."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT processing_status FROM processing_log WHERE file_name = ? AND processing_status = 'success'",
        (file_name,)
    )
    result = cursor.fetchone()
    conn.close()
    return result is not None


def log_processing(file_name: str, file_size: int, status: str, error_message: str = None,
                   records_filtered: int = 0, records_inserted: int = 0):
    """Log file processing status with filtering statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO processing_log 
        (file_name, file_size, processing_status, processing_time, error_message, records_filtered, records_inserted)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (file_name, file_size, status, datetime.now(timezone.utc), error_message, records_filtered, records_inserted))
    conn.commit()
    conn.close()


def extract_glm_data(file_path: str) -> Dict[str, Any]:
    """
    Extract data from GLM NetCDF file.

    Args:
        file_path: Path to NetCDF file

    Returns:
        Dictionary containing extracted data
    """
    dataset = None
    try:
        dataset = nc.Dataset(file_path, 'r')
        data = {}

        # Extract events data
        if 'event_id' in dataset.variables:
            data['events'] = {
                'event_id': dataset.variables['event_id'][:].copy(),
                'event_time_offset': dataset.variables['event_time_offset'][:].copy(),
                'event_lat': dataset.variables['event_lat'][:].copy(),
                'event_lon': dataset.variables['event_lon'][:].copy(),
                'event_energy': dataset.variables['event_energy'][:].copy(),
            }

        # Extract groups data
        if 'group_id' in dataset.variables:
            data['groups'] = {
                'group_id': dataset.variables['group_id'][:].copy(),
                'group_time_offset': dataset.variables['group_time_offset'][:].copy(),
                'group_lat': dataset.variables['group_lat'][:].copy(),
                'group_lon': dataset.variables['group_lon'][:].copy(),
                'group_area': dataset.variables['group_area'][:].copy(),
                'group_energy': dataset.variables['group_energy'][:].copy(),
                'group_quality_flag': dataset.variables['group_quality_flag'][:].copy(),
            }

        # Extract flashes data
        if 'flash_id' in dataset.variables:
            data['flashes'] = {
                'flash_id': dataset.variables['flash_id'][:].copy(),
                'flash_time_offset_of_first_event': dataset.variables['flash_time_offset_of_first_event'][:].copy(),
                'flash_time_offset_of_last_event': dataset.variables['flash_time_offset_of_last_event'][:].copy(),
                'flash_lat': dataset.variables['flash_lat'][:].copy(),
                'flash_lon': dataset.variables['flash_lon'][:].copy(),
                'flash_area': dataset.variables['flash_area'][:].copy(),
                'flash_energy': dataset.variables['flash_energy'][:].copy(),
                'flash_quality_flag': dataset.variables['flash_quality_flag'][:].copy()
            }

        return data

    except Exception as e:
        logger.error(f"Error extracting data from {file_path}: {e}")
        return {}
    finally:
        # Ensure dataset is properly closed
        if dataset is not None:
            try:
                dataset.close()
            except:
                pass


def insert_data_to_db(file_name: str, data: Dict[str, Any]) -> tuple[int, int]:
    """
    Insert extracted data into SQLite database.
    Returns (records_filtered, records_inserted) for logging.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    records_filtered = 0
    records_inserted = 0

    try:
        # Calculate original record counts for filtering stats
        original_events = len(data.get('events', {}).get('event_id', []))
        original_groups = len(data.get('groups', {}).get('group_id', []))
        original_flashes = len(data.get('flashes', {}).get('flash_id', []))

        # Apply geographic filtering
        filtered_data = _filter_data_by_bounds(data, DATA_BOUNDS)

        # Calculate filtered record counts
        filtered_events = len(filtered_data.get('events', {}).get('event_id', []))
        filtered_groups = len(filtered_data.get('groups', {}).get('group_id', []))
        filtered_flashes = len(filtered_data.get('flashes', {}).get('flash_id', []))

        records_filtered = (original_events + original_groups + original_flashes) - \
                           (filtered_events + filtered_groups + filtered_flashes)

        # Insert events data
        if 'events' in filtered_data and filtered_events > 0:
            events_df = pd.DataFrame(filtered_data['events'])
            events_df['file_name'] = file_name
            events_df.to_sql('glm_events', conn, if_exists='append', index=False,
                             dtype={
                                 'event_id': 'REAL',
                                 'event_time_offset': 'REAL',
                                 'event_lat': 'REAL',
                                 'event_lon': 'REAL',
                                 'event_energy': 'REAL',
                                 'parent_group_id': 'REAL'
                             })
            logger.info(f"Inserted {len(events_df)} events from {file_name}")
            records_inserted += len(events_df)

        # Insert groups data
        if 'groups' in filtered_data and filtered_groups > 0:
            groups_df = pd.DataFrame(filtered_data['groups'])
            groups_df['file_name'] = file_name
            groups_df.to_sql('glm_groups', conn, if_exists='append', index=False,
                             dtype={
                                 'group_id': 'REAL',
                                 'group_time_offset': 'REAL',
                                 'group_lat': 'REAL',
                                 'group_lon': 'REAL',
                                 'group_area': 'REAL',
                                 'group_energy': 'REAL',
                                 'group_quality_flag': 'INTEGER',
                                 'parent_flash_id': 'REAL'
                             })
            logger.info(f"Inserted {len(groups_df)} groups from {file_name}")
            records_inserted += len(groups_df)

        # Insert flashes data
        if 'flashes' in filtered_data and filtered_flashes > 0:
            flashes_df = pd.DataFrame(filtered_data['flashes'])
            flashes_df['file_name'] = file_name
            flashes_df.to_sql('glm_flashes', conn, if_exists='append', index=False,
                              dtype={
                                  'flash_id': 'REAL',
                                  'flash_time_offset': 'REAL',
                                  'flash_lat': 'REAL',
                                  'flash_lon': 'REAL',
                                  'flash_area': 'REAL',
                                  'flash_energy': 'REAL',
                                  'flash_quality_flag': 'INTEGER'
                              })
            logger.info(f"Inserted {len(flashes_df)} flashes from {file_name}")
            records_inserted += len(flashes_df)

        conn.commit()

        if records_filtered > 0:
            logger.info(f"Filtered out {records_filtered} records outside geographic bounds for {file_name}")

        return records_filtered, records_inserted

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def get_data_summary() -> Dict[str, int]:
    """Get summary of data in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    summary = {}

    # Count records in each table
    for table in ['glm_events', 'glm_groups', 'glm_flashes', 'processing_log']:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        summary[table] = cursor.fetchone()[0]

    # Add filtering statistics
    cursor.execute(
        "SELECT SUM(records_filtered), SUM(records_inserted) FROM processing_log WHERE processing_status = 'success'")
    result = cursor.fetchone()
    if result and result[0] is not None:
        summary['total_records_filtered'] = result[0]
        summary['total_records_inserted'] = result[1]

    conn.close()
    return summary


def purge_old_data(max_age_hours: int = 12, vacuum: bool = True) -> dict:
    '''
    Delete rows older than max_age_hours based on created_at/processing_time.
    Returns counts deleted per table. Optionally VACUUMs to reclaim disk space.
    '''
    if max_age_hours <= 0:
        raise ValueError('max_age_hours must be > 0')

    cutoff_clause = f'-{int(max_age_hours)} hours'
    stats = {}

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        # children first
        cur.execute("DELETE FROM glm_events WHERE created_at < datetime('now', ?)", (cutoff_clause,))
        stats['glm_events_deleted'] = cur.rowcount

        cur.execute("DELETE FROM glm_groups WHERE created_at < datetime('now', ?)", (cutoff_clause,))
        stats['glm_groups_deleted'] = cur.rowcount

        cur.execute("DELETE FROM glm_flashes WHERE created_at < datetime('now', ?)", (cutoff_clause,))
        stats['glm_flashes_deleted'] = cur.rowcount

        cur.execute("DELETE FROM processing_log WHERE processing_time < datetime('now', ?)", (cutoff_clause,))
        stats['processing_log_deleted'] = cur.rowcount

        conn.commit()
    finally:
        conn.close()

    if vacuum:
        try:
            conn2 = sqlite3.connect(DB_PATH)
            conn2.execute('VACUUM')
            conn2.execute('PRAGMA optimize')
            conn2.close()
            stats['vacuumed'] = True
        except Exception:
            stats['vacuumed'] = False

    logger.info(
        'Purged data older than %s: %s',
        cutoff_clause, ', '.join(f'{k}={v}' for k, v in stats.items())
    )
    return stats


class GLMProcessor:
    """Process GLM data from local files or AWS S3 to SQLite database."""

    def __init__(self, bucket_name: str = None, aws_profile: Optional[str] = None,
                 use_unsigned: bool = True, local_data_path: Optional[Path] = None):
        """
        Initialize GLM processor.

        Args:
            bucket_name: S3 bucket containing GLM data (optional if using local data)
            aws_profile: AWS profile name (optional)
            use_unsigned: Use unsigned requests for public buckets (no credentials needed)
            local_data_path: Path to local data directory (overrides environment variable)
        """
        # Set up local data path
        self.local_data_path = local_data_path or LOCAL_DATA_PATH

        if self.local_data_path:
            logger.info(f"Using local data source: {self.local_data_path}")
            self.s3_client = None
            self.bucket_name = None
        else:
            # Initialize AWS S3 client for remote data
            self.bucket_name = bucket_name

            if use_unsigned:
                # For public buckets - no credentials required
                from botocore import UNSIGNED
                from botocore.config import Config
                self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
                logger.info("Using unsigned requests for public S3 bucket")
            else:
                # For private buckets - use credentials
                session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
                self.s3_client = session.client('s3')

        # Initialize database
        init_database()

        # Log the geographic bounds being used
        if DATA_BOUNDS:
            min_lon, min_lat, max_lon, max_lat = DATA_BOUNDS
            logger.info(
                f"Geographic filtering enabled: lon=[{min_lon:.2f}, {max_lon:.2f}], lat=[{min_lat:.2f}, {max_lat:.2f}]")
        else:
            logger.info("No geographic filtering - ingesting global data")

    def list_local_files(self, prefix: str = 'GLM-L2-LCFA', max_files: int = 10) -> List[str]:
        """
        List local .nc files, returning the most recently modified ones.

        Args:
            max_files: Maximum number of files to return

        Returns:
            List of file paths sorted by modification time (newest first)
        """
        if not self.local_data_path or not self.local_data_path.exists():
            logger.warning(f"Local data path does not exist: {self.local_data_path}")
            return []

        try:
            # Find all .nc files recursively
            nc_files = []
            logger.info(f'Checking for GLM files with prefix: {prefix}')

            for nc_file in self.local_data_path.rglob("*.nc"):
                if nc_file.is_file() and prefix in nc_file.name.upper():
                    nc_files.append(nc_file)

            # Sort by modification time (newest first) and limit
            nc_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            nc_files = nc_files[:max_files]

            file_paths = [str(f) for f in nc_files]
            logger.info(f'Found {len(file_paths)} local GLM files')
            return file_paths

        except Exception as e:
            logger.error(f'Error listing local files: {e}')
            return []

    def list_s3_files(self, prefix: str = 'GLM-L2-LCFA/', max_files: int = 10, recent_hours: int = 24) -> list[str]:
        '''
        Return newest GLM .nc object keys by scanning recent hour prefixes only.
        '''
        if not self.s3_client:
            logger.error("S3 client not initialized - cannot list S3 files")
            return []

        try:
            prefixes = _recent_glm_prefixes(prefix, recent_hours)
            top_n: list[tuple[float, str]] = []  # (timestamp, key)
            push, replace = heapq.heappush, heapq.heapreplace

            paginator = self.s3_client.get_paginator('list_objects_v2')
            for pfx in prefixes:
                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=pfx):
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        if not (key.endswith('.nc') and 'GLM' in key.upper()):
                            continue
                        ts = obj['LastModified'].timestamp()
                        if len(top_n) < max_files:
                            push(top_n, (ts, key))
                        elif ts > top_n[0][0]:
                            replace(top_n, (ts, key))

                # cheap short-circuit: if we already have max_files after the most recent hour, older hours won't beat them
                if len(top_n) >= max_files and pfx == prefixes[0]:
                    break

            top_n.sort(key=lambda x: x[0], reverse=True)  # newest → oldest
            keys = [k for _, k in top_n]
            logger.info(f'Found {len(keys)} recent GLM files (scanned ~{min(recent_hours, len(prefixes))} hours).')
            return keys
        except Exception as e:
            logger.error(f'Error listing S3 files: {e}')
            return []

    def process_local_file(self, file_path: str) -> bool:
        """Process a local .nc file."""
        file_name = os.path.basename(file_path)

        if is_file_processed(file_name):
            logger.info(f'Skipping {file_name} - already processed')
            return True

        try:
            logger.info(f'Processing local file: {file_name}')

            # Get file size
            file_size = os.path.getsize(file_path)
            logger.info(f'Processing {file_name} ({file_size:,} bytes)')

            # Extract data directly from local file
            data = extract_glm_data(file_path)

            if data:
                records_filtered, records_inserted = insert_data_to_db(file_name, data)
                log_processing(file_name, file_size, 'success',
                               records_filtered=records_filtered, records_inserted=records_inserted)
                logger.info(
                    f'Successfully processed {file_name} - inserted {records_inserted} records, filtered {records_filtered}')
                return True
            else:
                log_processing(file_name, file_size, 'failed', 'No data extracted')
                logger.warning(f'No data extracted from {file_name}')
                return False

        except Exception as e:
            error_msg = str(e)
            log_processing(file_name, 0, 'error', error_msg)
            logger.error(f'Error processing {file_name}: {error_msg}')
            return False

    def process_s3_file(self, s3_key: str) -> bool:
        """Process an S3 file (original logic)."""
        file_name = os.path.basename(s3_key)

        if is_file_processed(file_name):
            logger.info(f'Skipping {file_name} - already processed')
            return True

        temp_path = None
        try:
            logger.info(f'Processing S3 file: {file_name}')

            import uuid
            temp_dir = tempfile.gettempdir()
            temp_filename = f'glm_temp_{uuid.uuid4().hex}.nc'
            temp_path = os.path.join(temp_dir, temp_filename)

            # Download file (no head_object)
            self.s3_client.download_file(self.bucket_name, s3_key, temp_path)

            # Get size from disk for logging
            file_size = os.path.getsize(temp_path)
            logger.info(f'Downloaded {file_name} ({file_size:,} bytes)')

            data = extract_glm_data(temp_path)

            if data:
                records_filtered, records_inserted = insert_data_to_db(file_name, data)
                log_processing(file_name, file_size, 'success',
                               records_filtered=records_filtered, records_inserted=records_inserted)
                logger.info(
                    f'Successfully processed {file_name} - inserted {records_inserted} records, filtered {records_filtered}')
                return True
            else:
                log_processing(file_name, file_size, 'failed', 'No data extracted')
                logger.warning(f'No data extracted from {file_name}')
                return False

        except Exception as e:
            error_msg = str(e)
            log_processing(file_name, 0, 'error', error_msg)
            logger.error(f'Error processing {file_name}: {error_msg}')
            return False
        finally:
            if temp_path and os.path.exists(temp_path):
                max_retries = 10
                for attempt in range(max_retries):
                    try:
                        import gc
                        gc.collect()
                        os.unlink(temp_path)
                        logger.debug(f'Successfully deleted temp file: {temp_path}')
                        break
                    except (OSError, PermissionError) as e:
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(0.2 * (attempt + 1))
                            logger.debug(f'Retry {attempt + 1} deleting temp file: {e}')
                        else:
                            logger.warning(f'Could not delete temp file {temp_path}: {e}')
                            try:
                                import uuid
                                cleanup_path = temp_path + f'.cleanup_{uuid.uuid4().hex}'
                                os.rename(temp_path, cleanup_path)
                                logger.info(f'Renamed temp file to {cleanup_path} for later cleanup')
                            except:
                                logger.warning(f'Temp file {temp_path} left in system - manual cleanup may be needed')

    def process_files(self, prefix: str = 'GLM-L2-LCFA/', max_files: int = 10, recent_hours: int = 6):
        """Process files from either local directory or S3."""

        if self.local_data_path:
            # Process local files
            files = self.list_local_files(max_files=max_files)
            if not files:
                logger.warning('No local GLM files found')
                return

            logger.info(f'Processing {len(files)} local files...')
            success_count = 0
            for i, file_path in enumerate(files, 1):
                logger.info(f'Processing local file {i}/{len(files)}: {os.path.basename(file_path)}')
                if self.process_local_file(file_path):
                    success_count += 1

        else:
            # Process S3 files (original logic)
            files = self.list_s3_files(prefix=prefix, max_files=max_files, recent_hours=recent_hours)
            if not files:
                logger.warning('No S3 GLM files found')
                return

            logger.info(f'Processing {len(files)} S3 files...')
            success_count = 0
            for i, s3_key in enumerate(files, 1):
                logger.info(f'Processing S3 file {i}/{len(files)}: {s3_key}')
                if self.process_s3_file(s3_key):
                    success_count += 1

        logger.info(f'Completed processing: {success_count}/{len(files)} files successful')


def main():
    """Main processing function that handles both local and S3 data sources."""

    if LOCAL_DATA_PATH:
        logger.info(f'Starting GLM ingest from local data: {LOCAL_DATA_PATH}')
        processor = GLMProcessor(local_data_path=LOCAL_DATA_PATH)
        # For local files, process all available files (don't limit by recent hours)
        processor.process_files(max_files=MAX_FILES)
    else:
        logger.info(f'Starting GLM ingest from S3: {BUCKET_NAME}')
        processor = GLMProcessor(bucket_name=BUCKET_NAME, use_unsigned=True)
        # For S3, use the normal time-based filtering
        processor.process_files(prefix=PREFIX, max_files=MAX_FILES, recent_hours=INGEST_RECENT_HOURS)

    # Print summary
    summary = get_data_summary()
    print("\nData Summary:")
    for table, count in summary.items():
        print(f"  {table}: {count:,} records")

    # Purge old data
    purge_old_data(max_age_hours=RETENTION_HOURS)


if __name__ == "__main__":
    while True:
        main()
        logger.info(f'Waiting {INGEST_REFRESH_SECONDS} seconds before refetching')
        time.sleep(INGEST_REFRESH_SECONDS)