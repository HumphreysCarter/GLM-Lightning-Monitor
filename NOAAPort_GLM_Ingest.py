import xarray
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

##############################################################################################################
# Settings

# Specify age of oldest flash
max_flash_age_min=30

# Specify GLM data path
glm_data_path='/home/humphreys/workspace/NOAAPort-GLM-Data/data/latest.nc'

# Specify output path for GLM plots/database
output_path='/home/humphreys/weather.carterhumphreys.com/glm/'

##############################################################################################################
# Util

def getGLM_Flash_Data(glm_file):
    glm_flash_data=pd.DataFrame(columns=['DateTime', 'Latitude', 'Longitude'])
    
    # Load GLM data from file
    glm_data = xarray.open_dataset(glm_file)

    # Read GLM dataset
    for flash_lat, flash_lon in zip(glm_data.flash_lat, glm_data.flash_lon):
        # Add flash to database
        glm_flash_data=glm_flash_data.append({'DateTime':glm_data.product_time.data, 
                                                  'Latitude':float(flash_lat.data), 
                                                  'Longitude':float(flash_lon.data)},                                         
                         ignore_index=True)
    glm_data.close()
    
    return glm_flash_data
    
##############################################################################################################
# Start Ingest Script

glm_flash_data=pd.DataFrame()

try:
    # Read in database
    glm_flash_data=pd.read_csv(f'{output_path}/glm_flash_data.csv', parse_dates=['DateTime'])
except:
    print('Building new database')

# Update flash database with new data
new_flash_data=getGLM_Flash_Data(glm_data_path)
glm_flash_data=glm_flash_data.append(new_flash_data)

# Remove old data from databases
max_data_time=datetime.utcnow()-timedelta(minutes=max_flash_age_min)
glm_flash_data=glm_flash_data.loc[glm_flash_data.DateTime >= max_data_time]

# Update data age
glm_flash_data['DataAge'] = datetime.utcnow()-glm_flash_data.DateTime
glm_flash_data['DataAge'] = [t/np.timedelta64(1, 'm') for t in glm_flash_data['DataAge'].values]

# Export database
glm_flash_data.to_csv(f'{output_path}/glm_flash_data.csv', index=False)
