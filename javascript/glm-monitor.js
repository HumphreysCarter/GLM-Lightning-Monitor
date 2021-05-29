
function getUrlParameter(sParam)
{
    var sPageURL = window.location.search.substring(1);
    var sURLVariables = sPageURL.split('&');
    for (var i = 0; i < sURLVariables.length; i++)
    {
        var sParameterName = sURLVariables[i].split('=');
        if (sParameterName[0] == sParam)
        {
            return sParameterName[1];
        }
    }
}

String.prototype.toTitleCase = function () {
    return this.replace(/\w\S*/g, function(txt){return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();});
};

// Update URL to store position
function updateURL(zoom, center) {
	history.replaceState( {} , '', 'glm-monitor.html?&lat=' + center.lat + '&lon=' + center.lng + '&z=' + zoom);
}

// Get URL view
var lat = getUrlParameter('lat');
var lon = getUrlParameter('lon');
var z = getUrlParameter('z');

// Set map center
var centerLat = 35.50987173838399
var centerLon = -111.90673828125001
var zoomLevel = 8

// Setup inital map
var map = L.map('map');
var overrideView = false;

// Set view to URL coordinates
if (lat != null && lon != null && z != null) {
	map.setView([lat, lon], z);

// Set to default
} else {
	map.setView([centerLat, centerLon], zoomLevel);
	overrideView = true;
}

// Update URL when map zoomed
	map.on('zoomend', function() {
        updateURL(map.getZoom(), map.getCenter());
    });

// Update URL when moved
map.on('moveend', function() {
    updateURL(map.getZoom(), map.getCenter());
});

// Add background OpenStreetMap
L.tileLayer('https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token=pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBndHRqcmZ3N3gifQ.rJcFIG214AriISLbB6B5aw', {
    maxZoom: 18,
    attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a> Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
    id: 'mapbox/streets-v11',
    tileSize: 512,
    zoomOffset: -1
}).addTo(map);

var icon = L.icon({
    iconUrl: '../bin/glm-icon.png',
    iconSize:     [47, 59.24], // size of the icon
    iconAnchor:   [22, 94], // point of the icon which will correspond to marker's location
    popupAnchor:  [-3, -76] // point from which the popup should open relative to the iconAnchor
});

// Get GLM GeoJSON
var geojsonLayer = new L.GeoJSON.AJAX('../bin/glm.json', {onEachFeature:popUp});

// Set GLM GeoJSON layer icons
function popUp(feature, layer) {
  layer.bindPopup("GLM Flash");
  layer.setIcon(icon);
}

// Add GLM GeoJSON layer to map
geojsonLayer.addTo(map);

// Add update time to page
fetch('../bin/glm_update.txt')
  .then(response => response.text())
  .then((data) => {
    document.getElementById("page_header").innerHTML = "&nbsp; Last Updated: " + data;
  })

// Reload page after 60 seconds
setTimeout(function () { location.reload(true); }, 60000);
