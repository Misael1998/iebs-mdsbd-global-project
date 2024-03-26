import os
import wget
from datetime import datetime, timedelta

import xarray as xr
import numpy as np
import cfgrib
import requests

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
