from geojson import Feature, Point, FeatureCollection
import xarray as xr

# Directories
data_path  = '/home/awips/glm/latest'
output_dir = '/home/awips/send2web/glm/'

# Open lightning dataset
ds = xr.open_dataset(data_path)

# Get lat/lon for each flash
flash_data = []
for lat, lon in zip(ds.flash_lat.data, ds.flash_lon.data):
    lon, lat = float(lon), float(lat)

    flash_data.append(Feature(geometry=Point((lon, lat))))

# Save data to GeoJSON
with open(f'{output_dir}/glm.json', 'w') as f:
    data = FeatureCollection(flash_data)
    f.write(str(data))

# Update file with time
with open(f'{output_dir}/glm_update.txt', 'w') as f:
    f.write(str(ds.product_time.data))
