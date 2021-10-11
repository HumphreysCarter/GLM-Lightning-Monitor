import xarray as xr
from glob import glob
from geojson import Feature, Point, FeatureCollection

# Directories
data_path    = '/data/goes/'
output_dir   = '/data/web/glm/'
num_of_files = 10

# Get GLM datasets
glmFiles = glob(f'{data_path}/OR_GLM-L2-LCFA_G16_*.nc')
print(glmFiles)
# Create empty list to store flash data
flash_data = []

# Get flash data for each file
updateTime = None
for i in range(num_of_files):
    if i < len(glmFiles):

        # Open lightning dataset
        ds = xr.open_dataset(glmFiles[-i])

        # Update product times
        if updateTime == None or updateTime <= ds.product_time.data:
            updateTime = ds.product_time.data

        # Get lat/lon for each flash
        for lat, lon in zip(ds.flash_lat.data, ds.flash_lon.data):
            lon, lat = float(lon), float(lat)

            flash_data.append(Feature(geometry=Point((lon, lat))))

# Save data to GeoJSON
with open(f'{output_dir}/glm.json', 'w') as f:
    data = FeatureCollection(flash_data)
    f.write(str(data))

# Update file with time
with open(f'{output_dir}/glm_update.txt', 'w') as f:
    f.write(str(updateTime))
