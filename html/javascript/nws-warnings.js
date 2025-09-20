// ====== NWS Warnings Module ======
// Requires: Leaflet (L) map instance and DOM elements to be available

class NWSWarnings {
    constructor(map, options = {}) {
        if (!map || typeof L === 'undefined') {
            throw new Error('Leaflet map instance is required');
        }

        this.map = map;
        this.options = {
            autoRefresh: true,
            refreshInterval: 300, // seconds
            apiBase: 'https://api.weather.gov',
            storagePrefix: 'nws:warnings:',
            ...options
        };

        // Layers
        this.warningsLayer = L.layerGroup().addTo(map);
        this.lastGeoJsonLayer = null;

        // State
        this.inflight = null;
        this.refreshTimer = null;
        this.enabled = true;
        this.visibleTypes = new Set();

        // DOM refs
        this.elements = {};

        // Warning types/colors
        this.warningTypes = {};
        this.colorsLoaded = false;

        // Prebind handlers used for Leaflet events
        this._onFeatureOver = this._onFeatureOver.bind(this);
        this._onFeatureOut = this._onFeatureOut.bind(this);
    }

    // --------- Setup ---------

    async init() {
        // no UI creation here
        // load warning types/colors
        await this.loadWarningTypes();

        // restore persisted settings (no DOM sync)
        this.restoreSettings();

        if (this.enabled && this.colorsLoaded) {
            this.startAutoRefresh();
            this.fetchWarnings();
        }
    }


    async loadWarningTypes() {
        try {
            const response = await fetch('config/common-data.json', {headers: {'Accept': 'application/json'}});
            if (!response.ok) throw new Error(`Failed to load common-data.json: ${response.status}`);

            const data = await response.json();
            const nwsColors = data?.fill;
            if (!nwsColors) throw new Error('No fill colors found in common-data.json');

            const priorityMap = NWSWarnings.PRIORITY_MAP;
            const warningTypes = {};

            Object.keys(nwsColors).forEach(name => {
                if (name === 'TEST') return;
                warningTypes[name] = {
                    color: nwsColors[name],
                    priority: priorityMap[name] ?? 0,
                    abbrev: NWSWarnings.createAbbrev(name)
                };
            });

            this.warningTypes = warningTypes;
            this.colorsLoaded = true;
            // console.log(`Loaded ${Object.keys(warningTypes).length} warning types from common-data.json`);
            return true;
        } catch (error) {
            console.error('Failed to load warning colors from common-data.json:', error);
            this.setStatus(`Warning: Could not load colors (${error.message})`, 'error');

            // Minimal fallback
            this.warningTypes = {
                'Tornado Warning': {color: '#FF0000', priority: 10, abbrev: 'TW'},
                'Severe Thunderstorm Warning': {color: '#FFA500', priority: 9, abbrev: 'SVW'},
                'Flash Flood Warning': {color: '#8B0000', priority: 8, abbrev: 'FFW'},
                'Special Weather Statement': {color: '#FFE4B5', priority: 0, abbrev: 'SWS'}
            };
            this.colorsLoaded = true;
            return false;
        }
    }

    // --------- Data ---------

    async fetchWarnings() {
        if (!this.enabled || !this.colorsLoaded) return;

        if (this.inflight) this.inflight.abort();
        this.inflight = new AbortController();
        this.setStatus('Loading warnings...');

        try {
            // All active alerts
            const url = `${this.options.apiBase}/alerts/active`;

            const response = await fetch(url, {
                signal: this.inflight.signal,
                headers: {
                    'User-Agent': 'GLM Lightning Monitor (https://github.com/HumphreysCarter/GLM-Lightning-Monitor)'
                }
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

            const data = await response.json();
            this.renderWarnings(data);

            const count = data.features?.length || 0;
            this.setStatus(`Loaded ${count} active warnings`, 'ok');
        } catch (error) {
            if (error.name === 'AbortError') return;
            console.error('Error fetching NWS warnings:', error);
            this.setStatus(`Error: ${error.message}`, 'error');
        } finally {
            this.inflight = null;
        }
    }

    renderWarnings(geojson) {
        this.clearWarnings();
        const features = geojson?.features || [];
        if (!features.length) return;

        // Filter to valid geometries and known types; render low priority first
        const valid = features
            .filter(f => f.geometry && this.getWarningConfig(f))
            .sort((a, b) => (this.getWarningConfig(a)?.priority ?? 0) - (this.getWarningConfig(b)?.priority ?? 0));

        const totalKnown = features.filter(f => this.getWarningConfig(f)).length;
        const mappable = valid.length;

        this.lastGeoJsonLayer = L.geoJSON(valid, {
            style: f => this.styleWarning(f),
            onEachFeature: (feature, layer) => {
                layer.bindPopup(this.createPopupContent(feature), {
                    maxWidth: 560,
                    minWidth: 340,
                    keepInView: true,
                    className: 'nws-popup-wide' // lets you target CSS
                });
                layer.on('mouseover', this._onFeatureOver);
                layer.on('mouseout', this._onFeatureOut);
            }
        }).addTo(this.warningsLayer);

        this.updateWarningVisibility();

        if (totalKnown !== mappable) {
            this.setStatus(`Loaded ${totalKnown} warnings (${mappable} mappable)`, 'ok');
        }
    }

    // --------- Feature helpers ---------

    setVisibleTypes(types) {
        // types: array of event names (strings); pass null/undefined to show all
        if (Array.isArray(types) && types.length) {
            this.visibleTypes = new Set(types);
        } else {
            this.visibleTypes = new Set(Object.keys(this.warningTypes));
        }
        this.updateWarningVisibility();
        this.saveSettings();
    }

    setOptions(opts = {}) {
        if ('autoRefresh' in opts) this.options.autoRefresh = !!opts.autoRefresh;
        if ('refreshInterval' in opts) this.options.refreshInterval = Number(opts.refreshInterval) || 300;
        if (this.options.autoRefresh && this.enabled) this.startAutoRefresh();
        this.saveSettings();
    }

    enable() {
        if (this.enabled) return;
        this.enabled = true;
        this.startAutoRefresh();
        this.fetchWarnings();
        this.saveSettings();
    }

    disable() {
        if (!this.enabled) return;
        this.enabled = false;
        this.stopAutoRefresh();
        this.clearWarnings();
        this.saveSettings();
    }


    _onFeatureOver(e) {
        e.target.setStyle({weight: 3, opacity: 0.9});
    }

    _onFeatureOut(e) {
        e.target.setStyle({weight: 2, opacity: 0.7});
    }

    getWarningConfig(feature) {
        const event = feature?.properties?.event;
        return event ? this.warningTypes[event] ?? null : null;
    }

    styleWarning(feature) {
        const cfg = this.getWarningConfig(feature);
        if (!cfg) return {opacity: 0, fillOpacity: 0};

        const isWatch = feature?.properties?.event?.includes('Watch');
        return {
            color: cfg.color,
            weight: 2,
            opacity: 0.7,
            fillColor: cfg.color,
            fillOpacity: 0.3,
            dashArray: isWatch ? '10,5' : null
        };
    }

    createPopupContent(feature) {
        const props = feature.properties || {};
        const cfg = this.getWarningConfig(feature);

        const headline = props.event || props.headline || 'Weather Alert';
        const description = props.description || 'No description available';
        const areas = props.areaDesc || 'Unknown area';
        const expires = props.expires ? new Date(props.expires).toLocaleString() : 'Unknown';

        const safe = (s, n) => String(s).substring(0, n);

        return `
    <div class='nws-popup warning-fade-in' style='--warn-color: ${cfg?.color || '#f59e0b'}'>
      <div class='nws-header'>
        <div class='nws-title'>${headline}</div>
      </div>

      <div class='nws-meta'>
        <div><strong>Areas:</strong> ${areas}</div>
        <div class='nws-times'>
          <span><strong>Expires:</strong> ${expires}</span>
        </div>
      </div>

      <div class='nws-section'>
        <div class='nws-body'>${safe(description, 1000)}${description.length > 1000 ? '…' : ''}</div>
      </div>

      ${props.instruction ? `
      <div class='nws-callout'>
        <div class='nws-callout-label'>Instructions</div>
        <div class='nws-callout-body'>${safe(props.instruction, 500)}${props.instruction.length > 500 ? '…' : ''}</div>
      </div>` : ''}
    </div>
  `;
    }

    updateWarningVisibility() {
        if (!this.lastGeoJsonLayer) return;
        this.lastGeoJsonLayer.eachLayer(layer => {
            const event = layer.feature?.properties?.event;
            const visible = event && this.visibleTypes.has(event);
            layer.setStyle({opacity: visible ? 0.7 : 0, fillOpacity: visible ? 0.3 : 0});
        });
    }

    clearWarnings() {
        this.warningsLayer.clearLayers();
        if (this.lastGeoJsonLayer) {
            this.lastGeoJsonLayer.remove();
            this.lastGeoJsonLayer = null;
        }
    }

    // --------- Refresh / Status ---------

    startAutoRefresh() {
        this.stopAutoRefresh();
        if (this.options.autoRefresh && this.enabled) {
            this.refreshTimer = setInterval(() => this.fetchWarnings(), this.options.refreshInterval * 1000);
        }
    }

    stopAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }

    setStatus(message, type = 'info') {
        // Optional callback for host app
        if (typeof this.options.onStatus === 'function') {
            this.options.onStatus({message, type});
        }
        // Otherwise, log quietly
        if (type === 'error') console.error('[NWS]', message);
        else if (type === 'ok') console.log('[NWS]', message);
        else console.debug('[NWS]', message);
    }


    // --------- Persistence ---------

    saveSettings() {
        try {
            const settings = {
                enabled: this.enabled,
                autoRefresh: this.options.autoRefresh,
                refreshInterval: this.options.refreshInterval,
                visibleTypes: Array.from(this.visibleTypes)
            };
            localStorage.setItem(`${this.options.storagePrefix}settings`, JSON.stringify(settings));
        } catch (e) {
            console.warn('Could not save NWS warnings settings:', e);
        }
    }

    restoreSettings() {
        try {
            const raw = localStorage.getItem(`${this.options.storagePrefix}settings`);
            if (!raw) return;
            const s = JSON.parse(raw);
            this.enabled = s.enabled ?? true;
            this.options.autoRefresh = s.autoRefresh ?? true;
            this.options.refreshInterval = s.refreshInterval ?? 300;
            this.visibleTypes = new Set(s.visibleTypes ?? Object.keys(this.warningTypes));
        } catch (e) {
            console.warn('Could not restore NWS warnings settings:', e);
        }
    }


    // --------- Public API ---------

    show() {
        this.enabled = true;
        this.startAutoRefresh();
        this.fetchWarnings();
        this.saveSettings();
    }

    hide() {
        this.enabled = false;
        this.stopAutoRefresh();
        this.clearWarnings();
        this.saveSettings();
    }

    destroy() {
        this.stopAutoRefresh();
        this.clearWarnings();
        if (this.inflight) this.inflight.abort();

        if (this.map && this.map.hasLayer && this.map.hasLayer(this.warningsLayer)) {
            this.map.removeLayer(this.warningsLayer);
        }

        const container = document.getElementById('warnings-container');
        if (container) container.remove();
    }

    // --------- Statics ---------

    static createAbbrev(name) {
        const make = (src, suffix) => src.replace(` ${suffix}`, '').split(' ').map(w => w[0]).join('') + suffix[0];
        if (name.includes('Warning')) return make(name, 'Warning');
        if (name.includes('Watch')) return make(name, 'Watch');
        if (name.includes('Advisory')) return make(name, 'Advisory');
        if (name.includes('Statement')) return make(name, 'Statement');
        return name.split(' ').map(w => w[0]).join('').substring(0, 3);
    }
}

// Priority map (higher is drawn on top)
NWSWarnings.PRIORITY_MAP = {
    // 10 - Life threatening immediate
    'Tornado Warning': 10,
    'Hurricane Warning': 10,
    'Tsunami Warning': 10,
    'Earthquake Warning': 10,
    'Typhoon Warning': 10,
    // 9 - Severe immediate threats
    'Severe Thunderstorm Warning': 9,
    'Flash Flood Warning': 9,
    'Extreme Wind Warning': 9,
    'Storm Surge Warning': 9,
    'Special Marine Warning': 9,
    // 8 - Major warnings
    'Blizzard Warning': 8,
    'Ice Storm Warning': 8,
    'Hurricane Force Wind Warning': 8,
    'Tropical Storm Warning': 8,
    'Storm Warning': 8,
    // 7 - Significant warnings
    'Flood Warning': 7,
    'Winter Storm Warning': 7,
    'High Wind Warning': 7,
    'Extreme Heat Warning': 7,
    'Extreme Cold Warning': 7,
    'Coastal Flood Warning': 7,
    'Lakeshore Flood Warning': 7,
    'High Surf Warning': 7,
    // 6 - Other warnings
    'Gale Warning': 6,
    'Snow Squall Warning': 6,
    'Red Flag Warning': 6,
    'Fire Warning': 6,
    'Avalanche Warning': 6,
    'Hard Freeze Warning': 6,
    'Freeze Warning': 6,
    'Heavy Freezing Spray Warning': 6,
    'Wind Chill Warning': 6,
    'Dust Storm Warning': 6,
    'Blowing Dust Warning': 6,
    'Lake Effect Snow Warning': 6,
    'Volcano Warning': 6,
    'Ashfall Warning': 6,
    // 5 - Minor warnings
    'Hazardous Seas Warning': 5,
    // 4 - Watches (high severity)
    'Tornado Watch': 4,
    'Hurricane Watch': 4,
    'Severe Thunderstorm Watch': 4,
    'Tsunami Watch': 4,
    'Hurricane Force Wind Watch': 4,
    'Extreme Heat Watch': 4,
    'Extreme Cold Watch': 4,
    // 3 - Watches (medium severity)
    'Flash Flood Watch': 3,
    'Blizzard Watch': 3,
    'Winter Storm Watch': 3,
    'Storm Surge Watch': 3,
    'Storm Watch': 3,
    'High Wind Watch': 3,
    'Gale Watch': 3,
    // 2 - Watches (lower severity)
    'Flood Watch': 2,
    'Coastal Flood Watch': 2,
    'Lakeshore Flood Watch': 2,
    'Fire Weather Watch': 2,
    'Hard Freeze Watch': 2,
    'Freeze Watch': 2,
    'Wind Chill Watch': 2,
    'Heavy Freezing Spray Watch': 2,
    'Lake Effect Snow Watch': 2,
    'Avalanche Watch': 2,
    'Hazardous Seas Watch': 2,
    // 1 - Advisories
    'Heat Advisory': 1,
    'Wind Advisory': 1,
    'High Wind Advisory': 1,
    'Winter Weather Advisory': 1,
    'Snow Advisory': 1,
    'Dense Fog Advisory': 1,
    'Wind Chill Advisory': 1,
    'Frost Advisory': 1,
    'Small Craft Advisory': 1,
    'Small Craft Advisory for Winds': 1,
    'Small Craft Advisory for Hazardous Seas': 1,
    'Small Craft Advisory for Rough Bar': 1,
    'Brisk Wind Advisory': 1,
    'Lake Wind Advisory': 1,
    'High Surf Advisory': 1,
    'Beach Hazards Statement': 1,
    'Coastal Flood Advisory': 1,
    'Lakeshore Flood Advisory': 1,
    'Flood Advisory': 1,
    'Small Stream Flood Advisory': 1,
    'Urban and Small Stream Flood Advisory': 1,
    'Hydrologic Advisory': 1,
    'Air Quality Alert': 1,
    'Air Stagnation Advisory': 1,
    'Dense Smoke Advisory': 1,
    'Dust Advisory': 1,
    'Blowing Dust Advisory': 1,
    'Freezing Fog Advisory': 1,
    'Freezing Rain Advisory': 1,
    'Freezing Spray Advisory': 1,
    'Hazardous Seas Advisory': 1,
    'Avalanche Advisory': 1,
    'Ashfall Advisory': 1,
    'Low Water Advisory': 1,
    'Tsunami Advisory': 1,
    // 0 - Statements and lowest priority
    'Special Weather Statement': 0,
    'Severe Weather Statement': 0,
    'Flash Flood Statement': 0,
    'Flood Statement': 0,
    'Coastal Flood Statement': 0,
    'Lakeshore Flood Statement': 0,
    'Marine Weather Statement': 0,
    'Hurricane Local Statement': 0,
    'Tropical Storm Local Statement': 0,
    'Tropical Depression Local Statement': 0,
    'Typhoon Local Statement': 0,
    'Rip Current Statement': 0,
    'Hazardous Weather Outlook': 0,
    'Hydrologic Outlook': 0,
    'Short Term Forecast': 0,
    'Special Avalanche Bulletin': 0,
    'Extreme Fire Danger': 0
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NWSWarnings;
}

// Global attachment for direct script include
if (typeof window !== 'undefined') {
    window.NWSWarnings = NWSWarnings;
}

export default NWSWarnings;