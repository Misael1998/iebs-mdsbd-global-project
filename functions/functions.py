import os
import warnings
import wget
from datetime import datetime, timedelta
from geopy import distance
from pystac_client import Client

import xarray as xr
import pandas as pd
import numpy as np
import cfgrib
import requests

import planetary_computer as pc
import rioxarray
import cv2

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
)

def get_url(date):
    """
    Given the date, construct an url to acces noaa-hrrr via aws s3 container
    """
    # Constants for creating the full URL
    blob_container = 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com'
    sector = "conus"
    date = datetime.strptime(date, "%Y-%m-%d").date()
    cycle = 1           # noon
    forecast_hour = 1   # offset from cycle time
    product = "wrfsfcf" # 2D surface levels
    
    # Put it all together
    file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
    url = f"{blob_container}/hrrr.{date:%Y%m%d}/{sector}/{file_path}"
    return url

def get_file(date):
    """
    Given the date, download noaa-hrrr file
    """
    attemps = 0
    path_or_date = date
    while attemps < 3:
        if attemps > 0:
            print(f'{date} : {attemps}')
        try:
            url = get_url(date)
            file_path = f'../data/downloaded/nrr/tmp/nrr_{date}.txt'
            resp = requests.get(url, timeout=600)
            f = open(file_path, 'wb')
            f.write(resp.content)
            f.close()
            attemps = 3
            path_or_date = file_path
        except:
            attemps = attemps + 1
            path_or_date = date

    return path_or_date

def get_ds(file_name):
    """
    Give the filename, return xarray dataset
    """
    return xr.open_dataset(
        file_name, 
        engine='cfgrib',
        backend_kwargs={'indexpath':''},
        filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'surface'}
    )

def get_features(ds, row):
    """
    Given xarray dataset and a row from the metadata, return dictionary with
    the features obtained from the noaa hrrr file
    """
    lat = row.latitude
    lon = row.longitude + 360
    
    abslat = np.abs(ds.latitude-lat)
    abslon = np.abs(ds.longitude-lon)
    c = np.maximum(abslon, abslat)
    
    ([xloc], [yloc]) = np.where(c == np.min(c))
    
    meta_info = {'uid': row['uid']}
    for varname, da in ds.sel(y=xloc, x=yloc).data_vars.items():
        meta_info[da.attrs['long_name']] = da.values.item()
    return meta_info

def gen_features(date, df, features, failed_dates):
    """
    Given the date, the whole metadata dataframe and a features list. Save the
    generated features in the features list
    """
    file_name = get_file(date)
    if file_name == date:
        failed_dates.append({'date': date})
    else:
        rows = df[df['date'] == date]
        ds = get_ds(file_name)
        for _, row in rows.iterrows():
            feature = get_features(ds, row)
            features.append(feature)
        os.remove(file_name)

def save_features_wrapper(args):
    """
    Save features wrapper for multiprocessesing
    """
    date, df, features, failed_dates = args
    try:
        gen_features(date, df, features, failed_dates)
    except:
        failed_dates.append({'date': date})

# get our bounding box to search latitude and longitude coordinates
def get_bounding_box(latitude, longitude, meter_buffer=3000):
    """
    Given a latitude, longitude, and buffer in meters, returns a bounding
    box around the point with the buffer on the left, right, top, and bottom.

    Returns a list of [minx, miny, maxx, maxy]
    """
    distance_search = distance.distance(meters=meter_buffer)

    # calculate the lat/long bounds based on ground distance
    # bearings are cardinal directions to move (south, west, north, and east)
    min_lat = distance_search.destination((latitude, longitude), bearing=180)[0]
    min_long = distance_search.destination((latitude, longitude), bearing=270)[1]
    max_lat = distance_search.destination((latitude, longitude), bearing=0)[0]
    max_long = distance_search.destination((latitude, longitude), bearing=90)[1]

    return [min_long, min_lat, max_long, max_lat]

# get our date range to search, and format correctly for query
def get_date_range(date, time_buffer_days=15):
    """Get a date range to search for in the planetary computer based
    on a sample's date. The time range will include the sample date
    and time_buffer_days days prior

    Returns a string"""
    datetime_format = "%Y-%m-%d"
    range_start = pd.to_datetime(date) - timedelta(days=time_buffer_days)
    date_range = f"{range_start.strftime(datetime_format)}/{pd.to_datetime(date).strftime(datetime_format)}"

    return date_range

def crop_sentinel_image(item, bounding_box, band):
    """
    Given a STAC item from Sentinel-2 and a bounding box tuple in the format
    (minx, miny, maxx, maxy), return a cropped portion of the item's visual
    imagery in the bounding box.

    Returns the image as a numpy array with dimensions (color band, height, width)
    """
    (minx, miny, maxx, maxy) = bounding_box

    image = rioxarray.open_rasterio(pc.sign(item.assets[band].href)).rio.clip_box(
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        crs="EPSG:4326",
    )

    return image.to_numpy()

# Get images
def get_images(row):
    """
    Given a row from the metadada, return 2 cropped images from sentinel
    1. True color image
    2. Water mask
    """
    bbox = get_bounding_box(row.latitude, row.longitude, meter_buffer=3000)
    date_range = get_date_range(row.date)
    
    # search the planetary computer sentinel-l2a and landsat level-2 collections
    search = catalog.search(
        collections=["sentinel-2-l2a"], 
        bbox=bbox, 
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 10}}
    )
    
    # get items
    items = [item for item in search.item_collection()]
    
    # get details of all of the items returned
    item_details = pd.DataFrame(
        [
            {
                "datetime": item.datetime.strftime("%Y-%m-%d"),
                "platform": item.properties["platform"],
                "min_long": item.bbox[0],
                "max_long": item.bbox[2],
                "min_lat": item.bbox[1],
                "max_lat": item.bbox[3],
                "bbox": item.bbox,
                "item_obj": item,
            }
            for item in items
        ]
    )
    
    # check which rows actually contain the sample location
    item_details["contains_sample_point"] = (
        (item_details.min_lat < row.latitude)
        & (item_details.max_lat > row.latitude)
        & (item_details.min_long < row.longitude)
        & (item_details.max_long > row.longitude)
    )
    item_details = item_details[item_details["contains_sample_point"]]
    item_details[["datetime", "platform", "contains_sample_point", "bbox"]].sort_values(
        by="datetime"
    )
    
    #Get best item
    best_item = (
    item_details[item_details.platform.str.contains("Sentinel")]
    .sort_values(by="datetime", ascending=False)
    .iloc[0]
    )
    item = best_item.item_obj

    bbox = get_bounding_box(row.latitude, row.longitude, meter_buffer=1000)
    true_color = crop_sentinel_image(item, bbox, "visual")
    scl = crop_sentinel_image(item, bbox, "SCL")
    
    # transpose
    visual = np.transpose(true_color, axes=[1, 2, 0]).astype(np.uint8)
    
    # Return images
    return visual, scl[0]

def get_sentinel_features(row):
    img, wm = get_images(row)
    water_scaled = np.stack([cv2.resize(wm, (img.shape[1], img.shape[0]))] * 3, -1) == 6
    f = {}
    if water_scaled.sum() == 0:
        f['uid'] = row.uid
        f['r'] = np.nan
        f['g'] = np.nan
        f['b'] = np.nan
        f['gmax'] = np.nan
        f['gmin'] = np.nan
        f['gvr'] = np.nan
        f['gvb'] = np.nan
        f['rvb'] = np.nan
        f['gmaxvb'] = np.nan
        f['gminvb'] = np.nan
    else:
        f['uid'] = row.uid
        f['r'] = img[:, :, 0][water_scaled[:, :, 0]].mean()
        f['g'] = img[:, :, 1][water_scaled[:, :, 1]].mean()
        f['b'] = img[:, :, 2][water_scaled[:, :, 2]].mean()
        f['gmax'] = np.percentile(img[:, :, 1][water_scaled[:, :, 1]], 95)
        f['gmin'] = np.percentile(img[:, :, 1][water_scaled[:, :, 1]], 5)
        f['gvr'] = f['g'] / f['r']
        f['gvb'] = f['g'] / f['b']
        f['rvb'] = f['r'] / f['b']
        f['gmaxvb'] = f['gmax'] / f['b']
        f['gminvb'] = f['gmin'] / f['b']
    
    return f

# Save Image
def save_sentinel_features(row, failed_points, features):
    """
    Given a row, save the features generated
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            features.append(get_sentinel_features(row))
    except Exception as e:
        failed_points.append({'uid': row['uid']})

def save_sentinel_features_wrapper(args):
    r, failed_points, features = args
    save_sentinel_features(r, failed_points, features)
