// ====== Config ======
const API_BASE = 'http://localhost:8000'; // change if API is elsewhere
const DEFAULT_AUTO_SEC = 60;              // auto-refresh cadence

// ====== DOM helpers ======
const $ = (id) => document.getElementById(id);

// Elements (must exist in HTML)
const minutesEl = $('minutes');
const autoEl = $('auto');
const intervalEl = $('interval');
const statusEl = $('status');
const refreshBtn = $('refresh');

// ====== Leaflet map ======
const map = L.map('map', {worldCopyJump: true}).setView([37.5, -96], 4);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 12,
    attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);
const layer = L.layerGroup().addTo(map);
let lastGeoJsonLayer = null;

// ====== State for rendering & timers ======
let inflight = null;           // AbortController for fetch
let refreshTimer = null;       // interval for fetching
let countdownTimer = null;     // per-second UI countdown
let nextRefreshTs = null;      // ms since epoch when next fetch will fire
let currentIntervalSec = DEFAULT_AUTO_SEC;

// keep these to make age-based styling consistent per render
let lastNowTs = Date.now();
let lastWindowMinutes = Number(minutesEl?.value || 30);

// ====== Status helpers (with countdown) ======
let lastStatusBase = 'Idle.';  // base message we append countdown to

// --- Persist / restore map view ---
const STORAGE_KEYS = {view: 'glm:view'};

function saveMapView() {
    try {
        const c = map.getCenter();
        const z = map.getZoom();
        localStorage.setItem(STORAGE_KEYS.view, JSON.stringify({
            lat: c.lat, lng: c.lng, zoom: z
        }));
    } catch {
    }
}

function restoreMapView() {
    try {
        const raw = localStorage.getItem(STORAGE_KEYS.view);
        if (!raw) return false;
        const v = JSON.parse(raw);
        if (Number.isFinite(v?.lat) && Number.isFinite(v?.lng) && Number.isFinite(v?.zoom)) {
            map.setView([v.lat, v.lng], v.zoom);
            return true;
        }
    } catch {
    }
    return false;
}

// === New: track what we've already rendered ===
let prevKeys = new Set();
let haveRenderedOnce = false;

function featureKey(feat) {
    const p = feat?.properties || {};
    // Pair file + flash_id; safer than id alone
    return `${p.source_file || ''}|${p.flash_id ?? ''}`;
}

function setBaseStatus(msg, kind = 'info') {
    lastStatusBase = msg;
    const color = kind === 'ok' ? 'ok' : kind === 'err' ? 'err' : '';
    statusEl.innerHTML = `<span class="${color}">${msg}</span>`;
}

function updateCountdownStatus() {
    if (!autoEl.checked || !nextRefreshTs) {
        // show base only when auto is off
        statusEl.innerHTML = `<span>${lastStatusBase}</span>`;
        return;
    }
    const remain = Math.max(0, Math.ceil((nextRefreshTs - Date.now()) / 1000));
    statusEl.innerHTML = `<span>${lastStatusBase} Next refresh in ${remain}s.</span>`;
}

// ====== BBox + styling helpers ======
function getBboxParam() {
    const b = map.getBounds();
    const minLat = b.getSouth(), minLon = b.getWest();
    const maxLat = b.getNorth(), maxLon = b.getEast();
    return `${minLon.toFixed(6)},${minLat.toFixed(6)},${maxLon.toFixed(6)},${maxLat.toFixed(6)}`;
}

// Age → color (0 min = red, window = blue)
function colorForAge(ageMin, windowMinutes) {
    const t = Math.max(0, Math.min(1, ageMin / Math.max(1, windowMinutes)));
    const hue = 0 + (220 * t); // 0=red → 220=blue
    return `hsl(${hue}, 100%, 50%)`;
}

// Age → size (newer ⇒ bigger)
function sizeForAge(ageMin, windowMinutes) {
    const t = Math.max(0, Math.min(1, ageMin / Math.max(1, windowMinutes))); // 0=new, 1=old
    const minR = 2;  // oldest
    const maxR = 8;  // newest
    return maxR - (maxR - minR) * t;
}

function styleForFeature(feat) {
    const iso = feat?.properties?.time_iso;
    let ageMin = lastWindowMinutes; // default = oldest
    if (iso) {
        const t = Date.parse(iso);
        if (!isNaN(t)) ageMin = Math.max(0, (lastNowTs - t) / 60000);
    }
    const fill = colorForAge(ageMin, lastWindowMinutes);
    const r = sizeForAge(ageMin, lastWindowMinutes);
    return {
        radius: r,
        weight: 0,
        opacity: 1,
        fillOpacity: 0.85,
        color: fill,
        fillColor: fill,
    };
}

function popupHtml(p) {
    const fmt = (v) => v == null ? '—' : v;
    let ageStr = '—';
    if (p.time_iso) {
        const t = Date.parse(p.time_iso);
        if (!isNaN(t)) ageStr = `${Math.max(0, (Date.now() - t) / 60000).toFixed(1)} min`;
    }
    return `
    <div><strong>Flash ${fmt(p.flash_id)}</strong></div>
    <div>Time (UTC): ${fmt(p.time_iso)}</div>
    <div>Age: ${ageStr}</div>
    <div>Energy: ${fmt(p.energy)}</div>
    <div>Area: ${fmt(p.area)}</div>
    <div>QF: ${fmt(p.quality_flag)}</div>
    <div style="color:#6b7280">${fmt(p.source_file)}</div>
  `;
}

// ====== Fetch & render ======
async function fetchAndRender() {
    // cancel any inflight fetch
    if (inflight) inflight.abort();
    const ctl = new AbortController();
    inflight = ctl;

    const minutes = Number(minutesEl.value || 30);
    const limit = 20000;
    const params = new URLSearchParams({minutes: String(minutes), limit: String(limit)});
    const bbox = getBboxParam();
    if (bbox) params.set('bbox', bbox);
    const url = `${API_BASE}/flashes?${params.toString()}`;

    // reset schedule if auto is on (so countdown restarts right after a fetch)
    if (autoEl.checked) {
        nextRefreshTs = Date.now() + currentIntervalSec * 1000;
    }

    setBaseStatus('Loading…');

    try {
        const t0 = performance.now();
        const res = await fetch(url, {signal: ctl.signal});
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const geojson = await res.json();

        // Capture render reference times
        lastNowTs = Date.now();
        lastWindowMinutes = minutes;

        // Build a fresh set of keys for this render
        const newKeys = new Set();

// Draw
        layer.clearLayers();
        if (lastGeoJsonLayer) lastGeoJsonLayer.remove();

        lastGeoJsonLayer = L.geoJSON(geojson, {
            pointToLayer: (feat, latlng) => {
                const key = featureKey(feat);
                // Only blink if we've rendered before AND this key wasn't seen last time
                const isNew = haveRenderedOnce && !prevKeys.has(key);

                const opts = {...styleForFeature(feat), className: isNew ? 'glm-flash glm-new' : 'glm-flash'};
                return L.circleMarker(latlng, opts);
            },
            onEachFeature: (feat, l) => {
                newKeys.add(featureKey(feat));
                l.bindPopup(popupHtml(feat.properties));
            }
        }).addTo(layer);

        // Swap the sets for next time; first load won't blink everything
        prevKeys = newKeys;
        haveRenderedOnce = true;

        const n = geojson.features?.length ?? 0;
        const dt = (performance.now() - t0).toFixed(0);
        setBaseStatus(`Rendered ${n.toLocaleString()} flashes in ${dt} ms.`, 'ok');
    } catch (err) {
        if (err.name === 'AbortError') return;
        console.error(err);
        setBaseStatus(`Error: ${err.message}`, 'err');
    } finally {
        inflight = null;
        // Immediately update status with countdown, if any
        updateCountdownStatus();
    }
}

// ====== Auto-refresh & countdown control ======
function startAuto() {
    stopAuto();
    currentIntervalSec = Math.max(2, Math.min(300, Number(intervalEl.value || DEFAULT_AUTO_SEC)));
    nextRefreshTs = Date.now() + currentIntervalSec * 1000;

    // fetch interval
    refreshTimer = setInterval(fetchAndRender, currentIntervalSec * 1000);

    // 1 Hz countdown updater
    countdownTimer = setInterval(updateCountdownStatus, 1000);

    updateCountdownStatus();
}

function stopAuto() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
    }
    if (countdownTimer) {
        clearInterval(countdownTimer);
        countdownTimer = null;
    }
    nextRefreshTs = null;
    setBaseStatus('Auto-refresh off.');
}

// ====== Wire up controls ======
refreshBtn.addEventListener('click', fetchAndRender);

autoEl.addEventListener('change', () => {
    if (autoEl.checked) startAuto(); else stopAuto();
});
intervalEl.addEventListener('change', () => {
    if (autoEl.checked) startAuto(); // restart with new cadence
});

// Refetch on map move if using bbox (debounced)
let moveDebounce = null;
map.on('moveend', () => {
    // always persist the latest view
    saveMapView();

    // refetch
    if (moveDebounce) clearTimeout(moveDebounce);
    moveDebounce = setTimeout(fetchAndRender, 250);
});

// ====== Initial boot ======
intervalEl.value = DEFAULT_AUTO_SEC;
autoEl.checked = true;

// restore view from previous session (if any) BEFORE first fetch
restoreMapView();

fetchAndRender();
startAuto();

