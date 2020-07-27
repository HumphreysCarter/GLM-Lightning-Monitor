"""

NOAAPort GLM Monitor

Copyright (c) 2020, Carter J. Humphreys All rights reserved.


"""


import io
import os
import os.path, time
import glob
import math
import pytz
import xarray
import smtplib, ssl
import numpy as np
import pandas as pd
import urllib.request as request
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt
from datetime import datetime, timedelta
from metpy.plots import USCOUNTIES
from metpy.units import units
from shapely.geometry import Point, Polygon
from urllib.request import urlopen, Request
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PIL import Image


##############################################################################################################

# Data Settings
max_flash_distance=100
max_flash_age_min=30
clear_time_min=30
local_timezone='US/Eastern' # Timezone list: https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568
distance_units='mile'

# Map settings
lat, lon = 43.11, -76.11
latitude_offset=0.75
longitude_offset=1.5
range_rings = [5, 10, 30]
flash_marker='+'
flash_size=100

# Notification settings
email_port = 465
smtp_server = 'server'
email_address='from@mail.com'
email_password='password'
send_to_emails=['send_to@mail.com']

# Specify GLM data directory
glm_data_path='data/'

# Specify output path for GLM plots/database
output_path='html/'


##############################################################################################################

# Average radius of the earth
earth_r=6371*units.km
    
def image_spoof(self, tile):
    req = Request(self._image_url(tile))
    req.add_header('User-agent','Anaconda 3')
    fh = urlopen(req) 
    im_data = io.BytesIO(fh.read())
    fh.close()
    return Image.open(im_data).convert(self.desired_tile_form), self.tileextent(tile), 'lower' # reformat for cartopy

def createRangeRing(lat, lon, distance, unit='km'):
    coords=[]
    lat=math.radians(lat)
    lon=math.radians(lon)
    distance=distance*units(unit)

    for hdg in range(0, 360):
        d=distance.to(earth_r.units)/earth_r
        hdg=(hdg/90)*math.pi/2
        lat2 = math.asin(math.sin(lat) * math.cos(d) + math.cos(lat) * math.sin(d) * math.cos(hdg))
        lon2 = lon + math.atan2(math.sin(hdg)*math.sin(d) * math.cos(lat),math.cos(d)-math.sin(lat)*math.sin(lat2))
        coords.append((math.degrees(lat2), math.degrees(lon2)))
    
    return Polygon(coords)

# Haversine formula to calculate the great-circle distance between two points
def distanceBetweenPoints(p1, p2, unit='km'):
    lat1, lon1 = math.radians(p1.x), math.radians(p1.y)
    lat2, lon2 = math.radians(p2.x), math.radians(p2.y)
    
    deltaLat = lat2-lat1
    deltaLon = lon2-lon1
    
    a = math.sin(deltaLat/2) * math.sin(deltaLat/2) + math.cos(lat1) * math.cos(lat2) * math.sin(deltaLon/2) * math.sin(deltaLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return (earth_r * c).to(units(unit))

# Initial bearing/forward azimuth for a straight line along a great-circle arc
def bearingBetweenPoints(p1, p2):
    lat1, lon1 = math.radians(p1.x), math.radians(p1.y)
    lat2, lon2 = math.radians(p2.x), math.radians(p2.y)
    
    y = math.sin(lon2-lon1) * math.cos(lat2);
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(lon2-lon1);
    theta = math.atan2(y, x)
    brng = (math.degrees(theta) + 360) % 360
    
    return brng

def getGLM_Flash_Data(glm_file):
    glm_flash_data=pd.DataFrame(columns=['DateTime', 'Latitude', 'Longitude', 'Distance', 'Direction'])
    
    # Load GLM data from file
    glm_data = xarray.open_dataset(glm_file)

    # Read GLM dataset
    for flash_lat, flash_lon in zip(glm_data.flash_lat, glm_data.flash_lon):

        # Calculate distance to GLM flash
        strike_distance=distanceBetweenPoints(Point(lat, lon), Point(flash_lat, flash_lon))
        strike_distance=strike_distance.to(distance_units)
        
        # Check if flash within max range
        if strike_distance <= max_flash_distance*units(distance_units):

            # Calculate direction to flash
            brng=bearingBetweenPoints(Point(lat, lon), Point(flash_lat, flash_lon))

            # Add flash to database    
            glm_flash_data=glm_flash_data.append({'DateTime':glm_data.product_time.data, 
                                                  'Latitude':flash_lat.data, 
                                                  'Longitude':flash_lon.data, 
                                                  'Distance':strike_distance.magnitude, 
                                                  'Direction':brng},                                         
                         ignore_index=True)
    glm_data.close()
    
    return glm_flash_data

def GetAreaFlashCount(glm_flash_data, coords, max_data_age=30):   
    # Filter to show only in time range
    data_age=timedelta(minutes=max_data_age)
    glm_flash_data=glm_flash_data.loc[glm_flash_data.DateTime >= datetime.utcnow()-data_age]
    
    # Check for lightning in range ring
    flash_count = 0
    for flash_lat, flash_lon in zip(glm_flash_data.Latitude, glm_flash_data.Longitude):
        if Point(flash_lat, flash_lon).within(coords):
            flash_count+=1
        
    return flash_count

def checkAlertStatus(range_rings, glm_flash_data, range_ring_trends):
    for range_ring in range_rings:

        curr_flash_count=range_ring_trends.iloc[-1][f'{range_ring}-{distance_units} Flash Count']
        prev_flash_count=range_ring_trends.iloc[-2][f'{range_ring}-{distance_units} Flash Count']

        # Lightning in ring ---> send alert
        if curr_flash_count > 0 and prev_flash_count == 0:
            telemetry=getFlashTelemetry(glm_flash_data)
            sendNotification(status='alert', ring=range_ring, telemetry=telemetry)
            

        # Lightning cleared ring ---> send all clear
        elif curr_flash_count == 0 and prev_flash_count > 0:
            sendNotification(status='clear', ring=range_ring)

def getFlashTelemetry(glm_flash_data):
    track=None
    bearing=None
    speed=None
    avg_distance=None
    min_distance=None
    
    # Sort by data age
    glm_flash_data=glm_flash_data.sort_values(['DataAge'])
    
    split=int(len(glm_flash_data)/2)
    
    newest = glm_flash_data.iloc[0:split].mean()
    oldest = glm_flash_data.iloc[split:].mean()
    
    # Determine track
    if newest.Distance < oldest.Distance:
        track='Approaching'
    elif newest.Distance > oldest.Distance:
        track='Moving Away'
    else:
        track='Stationary'
        
    # Determine bearing
    bearing=newest.Direction
    
    # Determine speed
    delta_time=newest.DataAge-oldest.DataAge
    delta_dist=newest.Distance-oldest.Distance
    speed=abs(delta_dist/(delta_time/60))
    
    # Determine average distance
    avg_distance=newest.Distance
    
    # Determine nearest flash
    min_distance=glm_flash_data.Distance.min()
    
    return (track, bearing, speed, avg_distance, min_distance)

# Send email notifications
def sendNotification(status, ring, telemetry=None):
    for receiver_email in send_to_emails:
        message = MIMEMultipart('alternative')
        message['Subject'] = 'GLM Lightning Notification'
        message['From'] = email_address
        message['To'] = receiver_email
        
        text=''
        time_local=pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone(local_timezone))
        if status == 'clear':      
            text=f'*** All Clear ***'
            text+=f'\n{ring}-{distance_units} radius of {lat}, {lon} clear for {clear_time_min}-min at {time_local.strftime("%I:%M %p %Z %a %b %d %Y")}.'
            
            
        elif status == 'alert':
            text+=f'*** Lightning Detected within {ring} {distance_units} of {lat}, {lon} ***'
            text+=f'\n{time_local.strftime("%I:%M %p %Z %a %b %d %Y")}'
            text+='\n'
            if telemetry!=None:
                track, bearing, speed, avg_distance, min_distance = telemetry
                text+=f'\n=== Motion Based on {max_flash_age_min}-min History==='
                text+=f'\n{track}'
                text+=f'\nAvg. Distance: {round(avg_distance, 1)} {distance_units}'
                text+=f'\nMin. Distance: {round(min_distance, 1)} {distance_units}'
                text+=f'\nBearing:  {round(bearing, 0)} deg'
                text+=f'\nSpeed:    {round(speed, 0)} {distance_units}/hr'
        
        message.attach(MIMEText(text, "plain"))

        # Create secure connection with server and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, email_port, context=context) as server:
            server.login(email_address, email_password)
            server.sendmail(email_address, receiver_email, message.as_string())

# Create map plot of GLM flashes
def makePlot(glm_flash_data, range_ring_coords):
    # Setup projection
    plotExtent = [lon-longitude_offset, lon+longitude_offset, lat-latitude_offset, lat+latitude_offset]
    proj = ccrs.LambertConformal(central_longitude=lon, central_latitude=lat)

    # Setup plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(plotExtent)

    # Plot range rings
    for coords in range_ring_coords:
        ring_lats, ring_lons = coords.exterior.xy
        plt.plot(ring_lons, ring_lats, c='k', linestyle='--', linewidth=2.5, transform=ccrs.PlateCarree())

    # Plot GLM flash data
    #size_scaler=flash_size*(1-(glm_flash_data.DataAge/max_flash_age_min))
    glm_plot=plt.scatter(glm_flash_data.Longitude, glm_flash_data.Latitude, c=glm_flash_data.DataAge, vmin=0, vmax=max_flash_age_min,
                s=flash_size, marker=flash_marker, label='GLM Flash', cmap='inferno_r', transform=ccrs.PlateCarree())
    
    # Add color bar
    cbar=plt.colorbar(glm_plot, orientation='horizontal', shrink=0.75, pad=0.05)
    cbar.set_label('Flash Age (Minutes)')
    
    # Add state borders
    state_borders=cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes', scale='10m', facecolor='none')
    ax.add_feature(state_borders, edgecolor='black', linewidth=2.0)

    # Add county borders:
    ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
     
    # Add map background
    cimgt.Stamen.get_image = image_spoof
    ax.add_image(cimgt.Stamen('terrain'), 9, interpolation='spline36')

    # Plot Title
    plt.title(f'GLM Flashes Last {max_flash_age_min}-min', loc='left')
    time_local=pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone(local_timezone))
    plt.title(f'Updated: {time_local.strftime("%I:%M %p %Z %a %b %d %Y")}', loc='right')

    plt.savefig(f'{output_path}/glm_map.png', bbox_inches='tight', dpi=100)
    plt.clf()
    plt.close()

# Create plot of GLM flash trends
def plotTrend(trend_data):
    
    # Setup plot
    trend_data.plot(figsize=(8, 10), kind='line', x='DateTime')
    
    plt.ylabel(f'Flashes Last {clear_time_min} Minutes')
    plt.xlabel('')
    plt.ylim(bottom=0)
    plt.xlim(left=datetime.utcnow()-timedelta(minutes=max_flash_age_min), right=datetime.utcnow()+timedelta(minutes=2))
    plt.grid()
        
    # Plot legend
    plt.legend(loc='upper left')

    # Plot Title
    plt.title(f'{max_flash_age_min}-min Lightning Trends', loc='left')
    time_local=pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone(local_timezone))
    plt.title(f'Updated: {time_local.strftime("%I:%M %p %Z %a %b %d %Y")}', loc='right')

    plt.savefig(f'{output_path}/glm_trend.png', bbox_inches='tight', dpi=100)
    plt.clf()
    plt.close()

##############################################################################################################
# Start script

# Read in databases
glm_flash_data=pd.read_csv(f'{output_path}/glm_flash_data.csv', parse_dates=['DateTime'])
range_ring_trends=pd.read_csv(f'{output_path}/range_ring_trends.csv', parse_dates=['DateTime'])

# Update flash database
glm_data_file = max(glob.glob(f'{glm_data_path}/OR_GLM_*.nc'), key=os.path.getctime)
new_flash_data=getGLM_Flash_Data(glm_data_file)
glm_flash_data=glm_flash_data.append(new_flash_data)

# Remove old data from databases
max_data_time=datetime.utcnow()-timedelta(minutes=max_flash_age_min)
glm_flash_data=glm_flash_data.loc[glm_flash_data.DateTime >= max_data_time]
range_ring_trends=range_ring_trends.loc[range_ring_trends.DateTime >= max_data_time]

# Update data age
glm_flash_data['DataAge'] = datetime.utcnow()-glm_flash_data.DateTime
glm_flash_data['DataAge'] = [t/np.timedelta64(1, 'm') for t in glm_flash_data['DataAge'].values]


# Build range rings
range_ring_coords = []
for range_ring in range_rings: 
    range_ring_coords.append(createRangeRing(lat, lon, range_ring, distance_units))

# Update lightning trends
current_trend={'DateTime':datetime.utcnow()}
for range_ring, coords in zip(range_rings, range_ring_coords):
    
    # Check for lightning in range rings
    flash_count=GetAreaFlashCount(glm_flash_data, coords, clear_time_min)
    
    # Save flash count to dataframe
    current_trend[f'{range_ring}-{distance_units} Flash Count']=flash_count

# Append new data
range_ring_trends=range_ring_trends.append(current_trend, ignore_index=True)

# Export databases
glm_flash_data.to_csv(f'{output_path}/glm_flash_data.csv', index=False)
range_ring_trends.to_csv(f'{output_path}/range_ring_trends.csv', index=False)

# Determine alert status
checkAlertStatus(range_rings, glm_flash_data, range_ring_trends)
    
# Make plots
plotTrend(range_ring_trends)
makePlot(glm_flash_data, range_ring_coords)