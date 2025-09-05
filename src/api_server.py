import re
import sqlite3
import time
from math import radians, sin, cos, asin, sqrt
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import threading
from contextlib import contextmanager

from .settings import DB_PATH

_MILES_PER_DEG_LAT = 69.0  # ~ miles per degree latitude
_MILES_PER_DEG_LON_EQ = 69.172  # ~ miles per deg longitude at equator

app = FastAPI(title='GLM Flash Data API', version='1.0.0', root_path='/api/')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Regex pulls sYYYYDDDHHMMSS (ignores any trailing digits, e.g., sub-seconds)
_S_RE = re.compile(r's(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})')

# Cache configuration
_CACHE_LOCK = threading.Lock()
_CACHE = {}
_CACHE_TTL_SECONDS = 30  # Cache data for 30 seconds
_CACHE_MAX_ENTRIES = 100


def _parse_start_utc_from_filename(file_name: str) -> Optional[datetime]:
    m = _S_RE.search(file_name)
    if not m:
        return None
    y, jjj, hh, mm, ss = m.groups()
    # %j = day-of-year (001–366)
    dt = datetime.strptime(f'{y}{jjj}{hh}{mm}{ss}', '%Y%j%H%M%S')
    return dt.replace(tzinfo=timezone.utc)


def _within_bbox(lon: float, lat: float, bbox: Tuple[float, float, float, float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)


def _row_to_feature(row: sqlite3.Row, now_utc: datetime) -> Optional[Dict[str, Any]]:
    file_name = row['file_name']
    base = _parse_start_utc_from_filename(file_name)
    if base is None:
        return None

    # Choose an offset: prefer first-event; else single offset; else None
    off_first = row['flash_time_offset_of_first_event']
    off_single = row['flash_time_offset']
    offset_s = off_first if off_first is not None else off_single
    if offset_s is None:
        return None

    t_utc = base + timedelta(seconds=float(offset_s))
    props = {
        'flash_id': row['flash_id'],
        'time_iso': t_utc.isoformat().replace('+00:00', 'Z'),
        'energy': row['flash_energy'],
        'area': row['flash_area'],
        'quality_flag': row['flash_quality_flag'],
        'source_file': file_name,
    }
    geom = {
        'type': 'Point',
        'coordinates': [float(row['flash_lon']), float(row['flash_lat'])],
    }
    return {'type': 'Feature', 'geometry': geom, 'properties': props}


@contextmanager
def _connect_with_retry(max_retries: int = 3, retry_delay: float = 0.1):
    """
    Context manager that handles database connection with retries and lock handling.
    """
    conn = None
    last_exception = None

    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10.0)  # 10 second timeout
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA busy_timeout=5000')  # 5 second busy timeout
            conn.execute('PRAGMA synchronous=NORMAL')  # Faster writes
            yield conn
            return
        except sqlite3.OperationalError as e:
            last_exception = e
            if conn:
                conn.close()
                conn = None

            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            else:
                raise e
        except Exception as e:
            if conn:
                conn.close()
            raise e
        finally:
            if conn:
                conn.close()


def _get_cache_key(endpoint: str, **params) -> str:
    """Generate cache key from endpoint and parameters."""
    sorted_params = sorted(params.items())
    param_str = "&".join(f"{k}={v}" for k, v in sorted_params)
    return f"{endpoint}?{param_str}"


def _get_from_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get data from cache if not expired."""
    with _CACHE_LOCK:
        if cache_key in _CACHE:
            cached_data, timestamp = _CACHE[cache_key]
            if time.time() - timestamp < _CACHE_TTL_SECONDS:
                return cached_data
            else:
                del _CACHE[cache_key]
    return None


def _set_cache(cache_key: str, data: Dict[str, Any]):
    """Store data in cache with timestamp."""
    with _CACHE_LOCK:
        # Simple LRU: remove oldest entries if cache is full
        if len(_CACHE) >= _CACHE_MAX_ENTRIES:
            oldest_key = min(_CACHE.keys(), key=lambda k: _CACHE[k][1])
            del _CACHE[oldest_key]

        _CACHE[cache_key] = (data, time.time())


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    '''Great-circle distance between two WGS84 points in miles.'''
    r = 3958.7613  # Earth radius in miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1r, lat2r = radians(lat1), radians(lat2)
    a = sin(dlat / 2) ** 2 + cos(lat1r) * cos(lat2r) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return r * c


def _deg_bbox_for_radius(lat: float, lon: float, miles: float) -> Tuple[float, float, float, float]:
    '''
    Approx bbox around (lat, lon) with given radius in miles.
    Returns (min_lon, min_lat, max_lon, max_lat).
    '''
    # Protect against cos(lat) ~ 0 near poles
    lon_deg_per_mile = _MILES_PER_DEG_LON_EQ * max(0.01, cos(radians(lat)))
    dlat = miles / _MILES_PER_DEG_LAT
    dlon = miles / lon_deg_per_mile
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


@app.get('/health')
def health():
    return {'ok': True}


@app.get('/flashes')
def get_flashes(
        minutes: Optional[int] = Query(None, ge=1, le=24 * 60,
                                       description='Minutes back from now (alternative to start_time/end_time)'),
        start_time: Optional[str] = Query(None, description='Start time in ISO format (e.g., 2024-01-01T12:00:00Z)'),
        end_time: Optional[str] = Query(None, description='End time in ISO format (e.g., 2024-01-01T13:00:00Z)'),
        bbox: Optional[str] = Query(None, description='minLon,minLat,maxLon,maxLat'),
        limit: int = Query(10000, ge=1, le=200000),
        use_cache: bool = Query(True, description='Use cached data if available'),
):
    '''
    Return GLM flashes as GeoJSON FeatureCollection.
    Time filtering options:
    1. Use `minutes` for last N minutes from now
    2. Use `start_time` and `end_time` for specific time range
    If both are provided, start_time/end_time takes precedence.
    Optional bbox filter: minLon,minLat,maxLon,maxLat
    '''

    # Validate time parameters
    if start_time is not None and end_time is not None:
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            if start_dt >= end_dt:
                raise HTTPException(status_code=400, detail='start_time must be before end_time')
            time_mode = 'range'
            window_start = start_dt
            window_end = end_dt
        except ValueError as e:
            raise HTTPException(status_code=400,
                                detail=f'Invalid time format. Use ISO format like 2024-01-01T12:00:00Z. Error: {str(e)}')
    elif start_time is not None or end_time is not None:
        raise HTTPException(status_code=400,
                            detail='Both start_time and end_time must be provided when using time range')
    elif minutes is not None:
        time_mode = 'minutes'
        now_utc = datetime.now(timezone.utc)
        window_start = now_utc - timedelta(minutes=minutes)
        window_end = now_utc + timedelta(minutes=5)  # Small buffer for clock drift
    else:
        raise HTTPException(status_code=400, detail='Either minutes or start_time/end_time must be provided')

    # Generate cache key including time mode
    if time_mode == 'range':
        cache_key = _get_cache_key('flashes', start_time=start_time, end_time=end_time, bbox=bbox, limit=limit)
    else:
        cache_key = _get_cache_key('flashes', minutes=minutes, bbox=bbox, limit=limit)

    # Try cache first if enabled
    if use_cache:
        cached_result = _get_from_cache(cache_key)
        if cached_result is not None:
            cached_result['_from_cache'] = True
            return cached_result

    try:
        bbox_tuple: Optional[Tuple[float, float, float, float]] = None
        if bbox:
            try:
                parts = [float(x) for x in bbox.split(',')]
                if len(parts) != 4:
                    raise ValueError
                bbox_tuple = (parts[0], parts[1], parts[2], parts[3])
            except Exception:
                raise HTTPException(status_code=400, detail='Invalid bbox format. Use minLon,minLat,maxLon,maxLat')

        # Try to get data from database with retries
        try:
            with _connect_with_retry() as conn:
                cur = conn.cursor()
                # For time range queries, we need a larger lookback to ensure we get all data
                if time_mode == 'range':
                    # Calculate how far back to look in the database
                    now_utc = datetime.now(timezone.utc)
                    lookback_hours = max(1, int((now_utc - window_start).total_seconds() / 3600) + 1)
                    lookback_str = f'-{lookback_hours} hours'
                else:
                    lookback_str = '-1 day'

                cur.execute(
                    '''
                    SELECT file_name,
                           flash_id,
                           flash_lat,
                           flash_lon,
                           flash_area,
                           flash_energy,
                           flash_quality_flag,
                           flash_time_offset_of_first_event,
                           flash_time_offset
                    FROM glm_flashes
                    WHERE created_at >= datetime('now', ?)
                    ''',
                    (lookback_str,)
                )
                rows = cur.fetchall()
        except sqlite3.OperationalError as db_error:
            # If database is still locked after retries, try to serve from cache regardless of expiration
            if "database is locked" in str(db_error).lower():
                with _CACHE_LOCK:
                    if cache_key in _CACHE:
                        cached_data, _ = _CACHE[cache_key]
                        cached_data['_from_stale_cache'] = True
                        cached_data['_cache_warning'] = 'Database unavailable, serving stale cache'
                        return cached_data

                # No cache available, return error with helpful message
                raise HTTPException(
                    status_code=503,
                    detail="Database temporarily unavailable and no cached data available. Please try again in a moment."
                )
            else:
                raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")

        features: List[Dict[str, Any]] = []
        now_utc = datetime.now(timezone.utc)

        for row in rows:
            feat = _row_to_feature(row, now_utc)
            if not feat:
                continue
            # time filter
            t_iso = feat['properties']['time_iso']
            t_utc = datetime.fromisoformat(t_iso.replace('Z', '+00:00'))
            if t_utc < window_start or t_utc > window_end:
                continue
            # bbox filter
            if bbox_tuple:
                lon, lat = feat['geometry']['coordinates']
                if not _within_bbox(lon, lat, bbox_tuple):
                    continue
            features.append(feat)
            if len(features) >= limit:
                break

        result = {
            'type': 'FeatureCollection',
            'features': features,
            '_from_cache': False,
            'time_range': {
                'start': window_start.isoformat(),
                'end': window_end.isoformat(),
                'mode': time_mode
            }
        }

        # Cache the result
        if use_cache:
            _set_cache(cache_key, result.copy())

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/stats/nearby')
def flashes_nearby(
        lat: float = Query(..., ge=-90, le=90, description='Center latitude'),
        lon: float = Query(..., ge=-180, le=180, description='Center longitude'),
        miles: float = Query(50.0, gt=0, le=2000, description='Search radius in miles'),
        minutes: int = Query(30, ge=1, le=24 * 60, description='Lookback window in minutes'),
        use_cache: bool = Query(True, description='Use cached data if available'),
):
    '''
    Return stats for flashes within `miles` of (lat, lon) over the last `minutes`.
    Stats: count, average distance (miles), min/max distance.
    '''
    # Generate cache key
    cache_key = _get_cache_key('nearby', lat=lat, lon=lon, miles=miles, minutes=minutes)

    # Try cache first if enabled
    if use_cache:
        cached_result = _get_from_cache(cache_key)
        if cached_result is not None:
            cached_result['_from_cache'] = True
            return cached_result

    try:
        now_utc = datetime.now(timezone.utc)
        window_start = now_utc - timedelta(minutes=minutes)

        # Reasonable default: just enough hours to cover the window, with a cushion
        scan_hours = max(1, min(24 * 7, int(minutes / 60) + 2))

        # Degree bbox prefilter
        min_lon, min_lat, max_lon, max_lat = _deg_bbox_for_radius(lat, lon, miles)

        # Try to get data from database with retries
        try:
            with _connect_with_retry() as conn:
                cur = conn.cursor()
                cur.execute(
                    '''
                    SELECT file_name,
                           flash_id,
                           flash_lat,
                           flash_lon,
                           flash_area,
                           flash_energy,
                           flash_quality_flag,
                           flash_time_offset_of_first_event,
                           flash_time_offset
                    FROM glm_flashes
                    WHERE flash_lat BETWEEN ? AND ?
                      AND flash_lon BETWEEN ? AND ?
                      AND created_at >= datetime('now', ?)
                    ''',
                    (min_lat, max_lat, min_lon, max_lon, f'-{scan_hours} hours')
                )
                rows = cur.fetchall()
        except sqlite3.OperationalError as db_error:
            # If database is still locked after retries, try to serve from cache regardless of expiration
            if "database is locked" in str(db_error).lower():
                with _CACHE_LOCK:
                    if cache_key in _CACHE:
                        cached_data, _ = _CACHE[cache_key]
                        cached_data['_from_stale_cache'] = True
                        cached_data['_cache_warning'] = 'Database unavailable, serving stale cache'
                        return cached_data

                # No cache available, return error with helpful message
                raise HTTPException(
                    status_code=503,
                    detail="Database temporarily unavailable and no cached data available. Please try again in a moment."
                )
            else:
                raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")

        # Walk rows: compute event time & distance, then filter precisely
        count = 0
        sum_dist = 0.0
        min_dist = None
        max_dist = None

        for row in rows:
            base = _parse_start_utc_from_filename(row['file_name'])
            if base is None:
                continue
            off_first = row['flash_time_offset_of_first_event']
            off_single = row['flash_time_offset']
            offset_s = off_first if off_first is not None else off_single
            if offset_s is None:
                continue
            t_utc = base + timedelta(seconds=float(offset_s))

            # Time window filter
            if t_utc < window_start or t_utc > now_utc + timedelta(minutes=5):
                continue

            lat2 = float(row['flash_lat'])
            lon2 = float(row['flash_lon'])
            dist_mi = _haversine_miles(lat, lon, lat2, lon2)

            if dist_mi <= miles:
                count += 1
                sum_dist += dist_mi
                min_dist = dist_mi if (min_dist is None or dist_mi < min_dist) else min_dist
                max_dist = dist_mi if (max_dist is None or dist_mi > max_dist) else max_dist

        avg_dist = (sum_dist / count) if count > 0 else None

        result = {
            'center': {'lat': lat, 'lon': lon},
            'radius_miles': miles,
            'bbox': [min_lon, min_lat, max_lon, max_lat],
            'minutes': minutes,
            'count': count,
            'avg_distance_miles': avg_dist,
            'min_distance_miles': min_dist,
            'max_distance_miles': max_dist,
        }

        # Cache the fresh result from database for potential future fallback
        if use_cache:
            _set_cache(cache_key, result.copy())

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/cache/stats')
def cache_stats():
    '''Return cache statistics for monitoring.'''
    with _CACHE_LOCK:
        cache_size = len(_CACHE)
        cache_entries = []
        current_time = time.time()

        for key, (_, timestamp) in _CACHE.items():
            age_seconds = current_time - timestamp
            cache_entries.append({
                'key': key,
                'age_seconds': age_seconds,
                'expired': age_seconds >= _CACHE_TTL_SECONDS
            })

    return {
        'cache_size': cache_size,
        'max_entries': _CACHE_MAX_ENTRIES,
        'ttl_seconds': _CACHE_TTL_SECONDS,
        'entries': cache_entries
    }


@app.post('/cache/clear')
def clear_cache():
    '''Clear all cached data.'''
    with _CACHE_LOCK:
        cleared_count = len(_CACHE)
        _CACHE.clear()

    return {
        'cleared_entries': cleared_count,
        'message': 'Cache cleared successfully'
    }