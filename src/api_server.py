import re
import sqlite3
from math import radians, sin, cos, asin, sqrt
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .settings import DB_PATH

_MILES_PER_DEG_LAT = 69.0    # ~ miles per degree latitude
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

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    '''Great-circle distance between two WGS84 points in miles.'''
    r = 3958.7613  # Earth radius in miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1r, lat2r = radians(lat1), radians(lat2)
    a = sin(dlat / 2)**2 + cos(lat1r) * cos(lat2r) * sin(dlon / 2)**2
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
    minutes: int = Query(30, ge=1, le=24*60),
    bbox: Optional[str] = Query(None, description='minLon,minLat,maxLon,maxLat'),
    limit: int = Query(10000, ge=1, le=200000),
):
    '''
    Return GLM flashes from the last `minutes` minutes as GeoJSON FeatureCollection.
    Optional bbox filter: minLon,minLat,maxLon,maxLat
    '''
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

        now_utc = datetime.now(timezone.utc)
        window_start = now_utc - timedelta(minutes=minutes)

        conn = _connect()
        cur = conn.cursor()

        # Pull a superset (recent inserts) to keep it fast, then filter precisely in Python.
        # You can widen/narrow the created_at window if needed.
        cur.execute(
            '''
            SELECT file_name, flash_id, flash_lat, flash_lon, flash_area, flash_energy,
                   flash_quality_flag, flash_time_offset_of_first_event, flash_time_offset
            FROM glm_flashes
            WHERE created_at >= datetime('now', '-1 day')
            '''
        )
        rows = cur.fetchall()
        conn.close()

        features: List[Dict[str, Any]] = []
        for row in rows:
            feat = _row_to_feature(row, now_utc)
            if not feat:
                continue
            # time filter
            t_iso = feat['properties']['time_iso']
            t_utc = datetime.fromisoformat(t_iso.replace('Z', '+00:00'))
            if t_utc < window_start or t_utc > now_utc + timedelta(minutes=5):
                continue
            # bbox filter
            if bbox_tuple:
                lon, lat = feat['geometry']['coordinates']
                if not _within_bbox(lon, lat, bbox_tuple):
                    continue
            features.append(feat)
            if len(features) >= limit:
                break

        return {'type': 'FeatureCollection', 'features': features}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/stats/nearby')
def flashes_nearby(
    lat: float = Query(..., ge=-90, le=90, description='Center latitude'),
    lon: float = Query(..., ge=-180, le=180, description='Center longitude'),
    miles: float = Query(50.0, gt=0, le=2000, description='Search radius in miles'),
    minutes: int = Query(30, ge=1, le=24*60, description='Lookback window in minutes'),
):
    '''
    Return stats for flashes within `miles` of (lat, lon) over the last `minutes`.
    Stats: count, average distance (miles), min/max distance.
    '''
    try:
        now_utc = datetime.now(timezone.utc)
        window_start = now_utc - timedelta(minutes=minutes)

        # Reasonable default: just enough hours to cover the window, with a cushion
        scan_hours = max(1, min(24 * 7, int(minutes / 60) + 2))

        # Degree bbox prefilter
        min_lon, min_lat, max_lon, max_lat = _deg_bbox_for_radius(lat, lon, miles)

        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            '''
            SELECT file_name, flash_id, flash_lat, flash_lon, flash_area, flash_energy,
                   flash_quality_flag, flash_time_offset_of_first_event, flash_time_offset
            FROM glm_flashes
            WHERE flash_lat BETWEEN ? AND ?
              AND flash_lon BETWEEN ? AND ?
              AND created_at >= datetime('now', ?)
            ''',
            (min_lat, max_lat, min_lon, max_lon, f'-{scan_hours} hours')
        )
        rows = cur.fetchall()
        conn.close()

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

        return {
            'center': {'lat': lat, 'lon': lon},
            'radius_miles': miles,
            'bbox': [min_lon, min_lat, max_lon, max_lat],
            'minutes': minutes,
            'count': count,
            'avg_distance_miles': avg_dist,
            'min_distance_miles': min_dist,
            'max_distance_miles': max_dist,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))