"""

NOAAPort GLM Monitor

Copyright (c) 2020, Carter J. Humphreys All rights reserved.


"""
import io
import math
import pytz
import xarray
import smtplib, ssl
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
from shutil import copyfile
from datetime import datetime, timedelta
from metpy.plots import USCOUNTIES
from metpy.units import units
from shapely.geometry import Point, Polygon
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib.request import urlopen, Request
from PIL import Image


##############################################################################################################

# Data Settings
max_flash_distance=100
max_flash_age_min=30
clear_time_min=30
local_timezone='US/Eastern' # Timezone list: https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568
distance_units='mile'

# Map settings
lat, lon, city_name = 42.38, -76.87, 'Watkins Glen, NY'
latitude_offset=0.75
longitude_offset=1.5
range_rings = [5, 10, 30]
flash_marker='+'
flash_size=100

# Animation Settings
animation_duration_minutes=30

# Proximity settings
flash_proximity_time_min=30

# Notification settings
email_port = 465
smtp_server = 'mail.carterhumphreys.com'
email_address='lightning.notifications@carterhumphreys.com'
email_password='VQ.nFZl#c!x1'
send_to_emails=['carter.humphreys@mods-for-grx.com']

# Specify GLM data path
glm_data_path='/home/humphreys/workspace/NOAAPort-GLM-Data/data/latest.nc'

# Specify output path for GLM plots/database
output_path='/home/humphreys/weather.carterhumphreys.com/glm/'


##############################################################################################################

# Average radius of the earth
earth_r=6371*units.km

# Source: https://gist.github.com/RobertSudwarts/acf8df23a16afdb5837f
def degrees_to_cardinal(d):
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    ix = round(d / (360. / len(dirs)))
    return dirs[ix % len(dirs)]

# Creates range rings based on lat, lon and distance
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
                                                  'Latitude':float(flash_lat.data),
                                                  'Longitude':float(flash_lon.data),
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

    # Determine lightning frequency
    flash_freq=flash_count/max_data_age
    flash_freq_desc='NONE'

    if flash_freq > 0 and flash_freq < 1:
        flash_freq_desc='OCNL'
    elif flash_freq >= 1 and flash_freq <= 6:
        flash_freq_desc='FREQ'
    elif flash_freq > 6:
        flash_freq_desc='CONS'

    return flash_count, flash_freq, flash_freq_desc

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
    # Get current local time
    time_local=pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone(local_timezone))

    # Send message to log
    if status == 'clear':
        exportAlertList(time_local, f'{ring}-{distance_units} radius',  f'All Clear for {clear_time_min}-min')
    elif status == 'alert':
        exportAlertList(time_local, f'{ring}-{distance_units} radius',  'Lightning Detected')

    # Send email message
    for receiver_email in send_to_emails:
        message = MIMEMultipart('alternative')
        message['Subject'] = 'GLM Lightning Notification'
        message['From'] = email_address
        message['To'] = receiver_email

        text=''
        if status == 'clear':
            text=f'*** All Clear ***'
            text+=f'\n{ring}-{distance_units} radius of {city_name} clear for {clear_time_min}-min at {time_local.strftime("%I:%M %p %Z %a %b %d %Y")}.'

        elif status == 'alert':
            text+=f'*** Lightning Detected within {ring} {distance_units} of {city_name} ***'
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

        # Attached message
        message.attach(MIMEText(text, "plain"))

        # Create secure connection with server and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, email_port, context=context) as server:
            server.login(email_address, email_password)
            server.sendmail(email_address, receiver_email, message.as_string())

# Export file name list for animation
def exportFileNames():
    f = open(f'{output_path}/glm-list.txt', 'w')
    time=datetime.utcnow()-timedelta(minutes=animation_duration_minutes)
    while time <= datetime.utcnow():
        f.write(f'/glm/archive/glm_map_{time:%Y%m%d_%H%M}.png\n')
        time+=timedelta(minutes=1)
    f.close()

# Exports alerts to file
def exportAlertList(time, range, status):
    # Read in alert list
    alertlist=pd.read_csv(f'{output_path}/glm-alert-list.txt', parse_dates=['DateTime'])

    # Remove old data from database
    max_data_time=datetime.utcnow()-timedelta(hours=24)
    alertlist=alertlist.loc[alertlist.DateTime >= max_data_time]

    # Append new alert
    alertlist=alertlist.append({'DateTime':time.strftime('%I:%M %p %Z %a %b %d %Y'), 'Range':range, 'Status':status}, ignore_index=True)

    # Write to file
    alertlist.to_csv(f'{output_path}/glm-alert-list.txt', index=False)

# Plot alert list table
def alertListTable():
    # Read in alert list
    alertlist=pd.read_csv(f'{output_path}/glm-alert-list.txt', parse_dates=['DateTime'])
    alertlist=alertlist.sort_values('DateTime', ascending=False)

    # Setup plt and make table
    plt.figure(figsize=(9, 3))
    if len(alertlist) > 0:
        # Get cell text
        cell_text = []
        for row in range(len(alertlist)):
            cell_text.append(alertlist.iloc[row])

        # Add table
        plt.table(cellText=cell_text, cellLoc='left', loc='top')

    # Hide axes
    plt.axis('off')

    # Export image and close plot
    plt.savefig(f'{output_path}/glm_alert_table.png', bbox_inches='tight', dpi=100)
    plt.clf()
    plt.close()


def image_spoof(self, tile): # this function pretends not to be a Python script
    url = self._image_url(tile) # get the url of the street map API
    req = Request(url) # start request
    req.add_header('User-agent','Anaconda 3') # add user agent to request
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read()) # get image
    fh.close() # close url
    img = Image.open(im_data) # open image with PIL
    img = img.convert(self.desired_tile_form) # set image format
    return img, self.tileextent(tile), 'lower' # reformat for cartopy

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
    glm_plot=plt.scatter(glm_flash_data.Longitude, glm_flash_data.Latitude, c=glm_flash_data.DataAge, vmin=0, vmax=max_flash_age_min,
                s=flash_size, marker=flash_marker, label='GLM Flash', cmap='plasma_r', transform=ccrs.PlateCarree())

    # Add color bar
    cbar=plt.colorbar(glm_plot, orientation='horizontal', shrink=0.75, pad=0.05)
    cbar.set_label('Flash Age (Minutes)')

    # Add state borders
    state_borders=cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes', scale='10m', facecolor='none')
    ax.add_feature(state_borders, edgecolor='black', linewidth=2.0)

    # Add county borders:
    ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)

    # Add map background
    cimgt.OSM.get_image = image_spoof # reformat web request for street map spoofing
    osm_img = cimgt.OSM() # spoofed, downloaded street map
    ax.add_image(osm_img, 9, interpolation='spline36', regrid_shape=2000) # add OSM with zoom specification


    # Plot Title
    plt.title(f'GLM Flashes Last {max_flash_age_min}-min', loc='left')
    time_local=pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone(local_timezone))
    plt.title(f'Updated: {time_local.strftime("%I:%M %p %Z %a %b %d %Y")}', loc='right')

    # Export image and close plot
    plt.savefig(f'{output_path}/glm_map.png', bbox_inches='tight', dpi=100)
    plt.clf()
    plt.close()

    # Copy file to archive
    copyfile(f'{output_path}/glm_map.png', f'{output_path}/archive/glm_map_{datetime.utcnow():%Y%m%d_%H%M}.png')

# Plot lightning frequency chart
def plotFreqChart(ring_freq, usePlainLanguage=False):
    colormap={'NONE':'green', 'OCNL':'yellow', 'FREQ':'orange', 'CONS':'red'}
    plain_language={'NONE':'NONE', 'OCNL':'OCCASIONAL', 'FREQ':'FREQUENT', 'CONS':'CONTINUOUS'}

    if len(range_rings) > 0:

       # Create figure and axes
       fig, axs = plt.subplots(1, len(range_rings), figsize=(10, 1))

       if len(range_rings) == 1:
           axs.set_title(f'{range_rings[0]}-{distance_units} Radius', fontsize=15)
           text=ring_freq[0]
           if usePlainLanguage:
              text=plain_language[text]
           axs.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', fontsize=20)
           axs.set_facecolor(colormap[ring_freq[0]])
           axs.get_xaxis().set_visible(False)
           axs.get_yaxis().set_visible(False)

       else:
           for ax, freq, d in zip(axs, ring_freq, range_rings):
               ax.set_title(f'{d}-{distance_units} Radius', fontsize=15)
               if usePlainLanguage:
                    freq=plain_language[freq]
               ax.text(0.5, 0.5, freq, horizontalalignment='center', verticalalignment='center', fontsize=20)
               ax.set_facecolor(colormap[freq])
               ax.get_xaxis().set_visible(False)
               ax.get_yaxis().set_visible(False)

    # Background
    fig.patch.set_alpha(0.0)

    # Export image and close plot
    plt.savefig(f'{output_path}/glm_freq_chart.png', bbox_inches='tight', dpi=50)
    plt.clf()
    plt.close()

# Plot proximity of lightning
def plotProximityChart(glm_flash_data, usePlainLanguage=False):

    # Get only flashes in last x minutes
    max_data_time=datetime.utcnow()-timedelta(minutes=flash_proximity_time_min)
    glm_flash_data=glm_flash_data.loc[glm_flash_data.DateTime >= max_data_time]

    # Only flashes within max range ring
    glm_flash_data=glm_flash_data.loc[glm_flash_data.Distance <= max(range_rings)]

    # Get minimum distance
    d=glm_flash_data.min().Distance
    proximity='NONE DETECTED'
    color='green'

    # Get proximity from distance
    if d <= 5:
        proximity='LTG OHD'
        if usePlainLanguage:
              proximity='LIGHTNING OVERHEAD'
        color='red'
    elif d > 5 and d <= 10:
        proximity='LTG VC'
        if usePlainLanguage:
              proximity='LIGHTNING IN VICINITY'
        color='orange'
    elif d > 10 and d <= 30:
        deg=int(glm_flash_data.mean().Direction)
        proximity=f'LTG DSNT {degrees_to_cardinal(deg)}'
        if usePlainLanguage:
              proximity=f'LIGHTNING DISTANT {degrees_to_cardinal(deg)}'
        color='yellow'

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 1))

    ax.set_title(f'Lightning Proximity', fontsize=15)
    ax.text(0.5, 0.5, proximity, horizontalalignment='center', verticalalignment='center', fontsize=20)
    ax.set_facecolor(color)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Background
    fig.patch.set_alpha(0.0)

    # Export image and close plot
    plt.savefig(f'{output_path}/glm_proximity.png', bbox_inches='tight', dpi=50)
    plt.clf()
    plt.close()


# Create plot of GLM flash trends
def plotTrend(trend_data):

    # Setup plot
    trend_data.plot(figsize=(8, 5), kind='line', x='DateTime')
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

    # Export image and close plot
    plt.savefig(f'{output_path}/glm_trend.png', bbox_inches='tight', dpi=100)
    plt.clf()
    plt.close()

##############################################################################################################
# Start script

# Read in databases
glm_flash_data=pd.read_csv(f'{output_path}/glm_flash_data.csv', parse_dates=['DateTime'])
range_ring_trends=pd.read_csv(f'{output_path}/range_ring_trends.csv', parse_dates=['DateTime'])

# Update flash database
new_flash_data=getGLM_Flash_Data(glm_data_path)
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
ring_freq=[]
for range_ring, coords in zip(range_rings, range_ring_coords):

    # Check for lightning in range rings
    flash_count, flash_freq, flash_freq_desc=GetAreaFlashCount(glm_flash_data, coords, clear_time_min)
    ring_freq.append(flash_freq_desc)

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
exportFileNames()
plotProximityChart(glm_flash_data)
plotFreqChart(ring_freq)
plotTrend(range_ring_trends)
alertListTable()
makePlot(glm_flash_data, range_ring_coords)
