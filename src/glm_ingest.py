import os
import time
import boto3
import heapq
import logging
import sqlite3
import tempfile
import netCDF4 as nc
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from .settings import DB_PATH, INGEST_REFRESH_SECONDS, BUCKET_NAME, PREFIX, MAX_FILES, INGEST_RECENT_HOURS, RETENTION_HOURS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
                       TEXT
                   )
                   ''')

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


def log_processing(file_name: str, file_size: int, status: str, error_message: str = None):
    """Log file processing status."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO processing_log 
        (file_name, file_size, processing_status, processing_time, error_message)
        VALUES (?, ?, ?, ?, ?)
    ''', (file_name, file_size, status, datetime.now(timezone.utc), error_message))
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


def insert_data_to_db(file_name: str, data: Dict[str, Any]):
    """Insert extracted data into SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Insert events data
        if 'events' in data and len(data['events']['event_id']) > 0:
            events_df = pd.DataFrame(data['events'])
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

        # Insert groups data
        if 'groups' in data and len(data['groups']['group_id']) > 0:
            groups_df = pd.DataFrame(data['groups'])
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

        # Insert flashes data
        if 'flashes' in data and len(data['flashes']['flash_id']) > 0:
            flashes_df = pd.DataFrame(data['flashes'])
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

        conn.commit()

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
    """Process GLM data from AWS S3 to SQLite database."""

    def __init__(self, bucket_name: str, aws_profile: Optional[str] = None,
                 use_unsigned: bool = True):
        """
        Initialize GLM processor.

        Args:
            bucket_name: S3 bucket containing GLM data
            db_path: Path to SQLite database file
            aws_profile: AWS profile name (optional)
            use_unsigned: Use unsigned requests for public buckets (no credentials needed)
        """
        self.bucket_name = bucket_name

        # Initialize AWS S3 client
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

    def list_s3_files(self, prefix: str = 'GLM-L2-LCFA/', max_files: int = 10, recent_hours: int = 24) -> list[str]:
        '''
        Return newest GLM .nc object keys by scanning recent hour prefixes only.
        '''
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

    def process_file(self, s3_key: str) -> bool:
        file_name = os.path.basename(s3_key)

        if is_file_processed(file_name):
            logger.info(f'Skipping {file_name} - already processed')
            return True

        temp_path = None
        try:
            logger.info(f'Processing {file_name}')

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
                insert_data_to_db(file_name, data)
                log_processing(file_name, file_size, 'success')
                logger.info(f'Successfully processed {file_name}')
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
        files = self.list_s3_files(prefix=prefix, max_files=max_files, recent_hours=recent_hours)
        if not files:
            logger.warning('No GLM files found')
            return
        logger.info(f'Processing {len(files)} files...')
        success_count = 0
        for i, s3_key in enumerate(files, 1):
            logger.info(f'Processing file {i}/{len(files)}: {s3_key}')
            if self.process_file(s3_key):
                success_count += 1
        logger.info(f'Completed processing: {success_count}/{len(files)} files successful')

def main():
    """Example usage of GLMProcessor."""

    # Initialize processor for public bucket (no credentials needed)
    logger.info(f'Starting GLM ingest for {BUCKET_NAME}')

    processor = GLMProcessor(bucket_name=BUCKET_NAME, use_unsigned=True)

    # Process files
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