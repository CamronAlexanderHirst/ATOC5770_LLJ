import numpy as np                  # For doing math
import matplotlib.pyplot as plt     # For plotting
import pandas as pd
import glob
import xarray as xr                 # For dealing with netCDF data
import os

"""
This code processes data_siteH_b1 lidar files and calculates:
- wind components
- wind speed
- wind direction

from 0 - 500m AGL. It then saves new dataframes with this information.

Author: Alex Hirst
Based on code by Professor Julie Lundquist
"""

max_altitude = 1000 # m
fig_path = "./plots/profiles/"

print('Done importing modules now')

path_to_data = "./data_siteH_b1"
save_path = "./data_siteH_alex/"
filenames = glob.glob(path_to_data+'/*.cdf')
#fig, ax = plt.subplots()

i = 0
# loop over files
for filename in filenames:
    i += 1
    print("Processing file {} of {}".format(i, len(filenames)))
    dataset = xr.open_dataset(filename, engine='netcdf4')
    df = dataset.to_dataframe()
    # plot the azimuth angle vs elevation angle
    azimuth_angles = []
    elevation_angles = []
    radial_velocities = []
    qc_radial_velocities = []
    measurement_altitudes = []
    for index, row in df.iterrows():
        # get angles and radial velocities
        azimuth_angles.append(row["azimuth"])
        elevation_angles.append(row["elevation"])
        radial_velocities.append(row["radial_velocity"])
        # get altitude of the measurement
        range = index[1]
        elevation = row["elevation"]
        measurement_altitude = range*np.cos(np.deg2rad(90-elevation))
        measurement_altitudes.append(measurement_altitude)  
    df["measurement_altitude"] = measurement_altitudes


    # Process data by altitude
    df = df[df['measurement_altitude'] <= max_altitude]  # only process data < max_altitude
    unique_altitudes = sorted(df['measurement_altitude'].unique())
    min_unique_azimuth = min(sorted(df['azimuth'].unique()))
    if min_unique_azimuth in [0.0, 23.75]:
        # only process data with elevation angle = 60
        unique_elevations = sorted(df['elevation'].unique())
        if 60.0 in unique_elevations:
            df =  df[df['elevation'] == 60.0]
        else: 
            print("------ DF Rejected Elevation ------ {}".format(unique_elevations))
            continue
    else:
        print("------ DF Rejected Azimuth ------ {}".format(min_unique_azimuth))
        continue

    times = []
    u_values = []
    v_values = []
    w_values = []
    horizontal_windspeeds = []
    horizontal_wind_directions = []
    altitudes = []
    elevations = []
    # re-create unique altitudes
    unique_altitudes = sorted(df['measurement_altitude'].unique())
    for ua in unique_altitudes:
        # pick out rows at desired altitude
        df_altitude = df[df['measurement_altitude'] == ua]

        # pick out north, east, south, west velocities and elevation angles
        if min(sorted(df_altitude['azimuth'].unique())) == 0.0:
            time = df_altitude[df_altitude['azimuth'] == 0].index[0]
            elevation = df_altitude[df_altitude['azimuth'] == 0]['elevation'].values[0]
            Vr_north = df_altitude[df_altitude['azimuth'] == 0]['radial_velocity'].values[0]
            Vr_east = df_altitude[df_altitude['azimuth'] == 90]['radial_velocity'].values[0]
            Vr_south = df_altitude[df_altitude['azimuth'] == 180]['radial_velocity'].values[0]
            Vr_west = df_altitude[df_altitude['azimuth'] == 270]['radial_velocity'].values[0]
        elif min(sorted(df_altitude['azimuth'].unique())) == 23.75:
            time = df_altitude[df_altitude['azimuth'] == 23.75].index[0]
            elevation = df_altitude[df_altitude['azimuth'] == 23.75]['elevation'].values[0]
            Vr_north = df_altitude[df_altitude['azimuth'] == 23.75]['radial_velocity'].values[0]
            Vr_east = df_altitude[df_altitude['azimuth'] == 113.75]['radial_velocity'].values[0]
            Vr_south = df_altitude[df_altitude['azimuth'] == 203.75]['radial_velocity'].values[0]
            Vr_west = df_altitude[df_altitude['azimuth'] == 293.75]['radial_velocity'].values[0]
        else:
            raise ValueError("Unknown azimuth - filtering broken.")

        # calculate u, v, w [see HW5 for equations]
        u = (Vr_east - Vr_west)/(2*np.sin(np.deg2rad(90-elevation)))
        v = (Vr_north - Vr_south)/(2*np.sin(np.deg2rad(90-elevation)))
        w = (Vr_north + Vr_south + Vr_east + Vr_west) / (4 * np.cos(np.deg2rad(90-elevation)))

        # rotate wind u and v components if azimuth is not aligned with north
        if min(sorted(df_altitude['azimuth'].unique())) == 23.75:
            theta = np.deg2rad(23.75)
            R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            rot_vect = np.matmul(R, [u, v])
            u, v = rot_vect[0], rot_vect[1]

        # calculate horizontal windspeed and direction
        horizontal_ws = np.sqrt(u**2 + v**2)
        horizontal_wd = np.arctan2(u, v)+np.pi  # east (u) over north (v)

        # save off data
        times.append(time)
        altitudes.append(ua)
        u_values.append(u)
        v_values.append(v)
        w_values.append(w)
        horizontal_windspeeds.append(horizontal_ws)
        horizontal_wind_directions.append(horizontal_wd)
        elevations.append(elevation)

    alex_df = pd.DataFrame(
        list(
        zip(u_values, 
            v_values, 
            w_values, 
            horizontal_windspeeds, 
            horizontal_wind_directions, 
            altitudes,
            elevations)),
        columns = ["u", 
                   "v", 
                   "w", 
                   "horizontal_windspeed", 
                   "horizontal_wind_direction", 
                   "altitude",
                   "elevations"],      
    )
    date_time = filename.split(".")
    time = date_time[-2]
    date = date_time[-3]

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(horizontal_windspeeds, altitudes)
    plt.xlabel("Windspeed [m/s]")
    plt.ylabel("Altitude [m]")
    plt.subplot(1,2,2)
    plt.plot(np.rad2deg(horizontal_wind_directions), altitudes)
    plt.xlabel("Wind Directions [deg]")
    plt.ylabel("Altitude [m]")
    plt.savefig(fig_path+str(date)+"_"+str(time)+".png", dpi=500, format="png")
    plt.close()
    alex_df.to_pickle(save_path+"alex_{}_{}.pkl".format(date, time))