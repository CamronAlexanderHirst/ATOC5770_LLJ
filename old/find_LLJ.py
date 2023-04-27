import pandas as pd
import glob 
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from datetime import datetime
from dateutil import tz

def find_classification(max_wspd, shear):
    '''
    Find the LLJ classification based on these values.
    
    From Vanderwende: 
            Speed    Shear
    LLJ-0:   10        5
    LLJ-1:   12        6
    LLJ-2:   16        8
    LLJ-3:   20       10
    '''
    
    if ((max_wspd>=20) and (shear>=10)):
        return 3
    elif ((max_wspd>=16) and (shear>=8)):
        return 2
    elif ((max_wspd>=12) and (shear>=6)):
        return 1
    else:
        return 0
    
def datestr(num,format="%Y-%m-%d %H:%M:%S.%f"):
    from datetime import datetime
    string=datetime.utcfromtimestamp(num).strftime(format)
    return string

source='/Users/alexhirst/Documents/CU Boulder/ATOC 5770/project/data_siteH_b1'
output_file_path = 'summary2.xlsx'
height_thresh = 700 # threshold for highest nose height to accept
ws_thresh = 10
shear_thresh = 5

# Read in data
files = glob.glob(source+'/*.nc')

summary_full = pd.DataFrame()

for file in files:
    df = xr.open_dataset(file)
    # set up output columns and dataframe
    LLJ_class = []
    nose_heights = []
    sfc_nose_shr = []
    above_nose_shr = []
    nose_wd = []
    nose_ws = []

    summary = pd.DataFrame()
    
    # loop through each time in the file
    for i in range(len(df.time)):
        df_i = df.isel(time=i)
        # skip if there isn't enough non-Nan values (maybe consider another approach?)
        if len(df_i.ws.dropna(dim='height')) < 35:
            LLJ_class.append(np.nan)
            nose_heights.append(np.nan)
            sfc_nose_shr.append(np.nan)
            above_nose_shr.append(np.nan)
            nose_wd.append(np.nan)
            nose_ws.append(np.nan)
            continue
        # find max windspeed and shear above nose
        ws_max = df_i.ws.max().values
        ws_max_idx = df_i.ws.argmax().values
        shear_above = np.ptp(df_i.ws[ws_max_idx:].dropna(dim='height').values)
        height = df_i.height[ws_max_idx].values
        # Check LLJ conditions
        if (ws_max > ws_thresh) and (shear_above > shear_thresh) and (height < height_thresh):
            shear_below = np.ptp(df_i.ws[:ws_max_idx].dropna(dim='height').values)
            class_ = find_classification(ws_max, shear_above)
            # add values to columns
            LLJ_class.append(class_)
            nose_heights.append(height)
            sfc_nose_shr.append(shear_below)
            above_nose_shr.append(shear_above)
            nose_wd.append(df_i.wd[ws_max_idx].values)
            nose_ws.append(ws_max)    
        else:
            LLJ_class.append(np.nan)
            nose_heights.append(np.nan)
            sfc_nose_shr.append(np.nan)
            above_nose_shr.append(np.nan)
            nose_wd.append(np.nan)
            nose_ws.append(np.nan)
    # Add columns to the dataframe
    to_zone = tz.gettz('America/Chicago')
    times = [datestr(tt) for tt in df.time.values]
    local_times = [pd.Timestamp(tt).tz_localize("UTC").tz_convert(to_zone) for tt in df.time.values()]
    summary['UTC Time (unix)'] = df.time.values
    summary['UTC Time'] = times
    summary['Local Time'] = local_times
    summary['LLJ class'] = LLJ_class
    summary['Nose Windspeed [m/s]'] = nose_ws
    summary['Nose Height [m]'] = nose_heights
    summary['Surface to nose shear [m/s]'] = sfc_nose_shr
    summary['Shear above nose [m/s]'] = above_nose_shr
    summary['Nose Wind Direction [degrees]'] = nose_wd
    # concatenate with the rest of the data
    summary_full = pd.concat([summary_full, summary])
    print(file, 'processed')
    
summary_full = summary_full.set_index(summary_full['UTC Time (unix)']).sort_index(ascending=True).drop(columns=['UTC Time (unix)'])

summary_full.to_excel(output_file_path)
