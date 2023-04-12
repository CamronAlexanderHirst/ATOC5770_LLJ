"""
This script conducts a VAD Fitting and does some plotting of LIDAR Data.
"""

import os
import csv
import time
import pickle
import numpy as np                  # For doing math
import matplotlib
import matplotlib.pyplot as plt     # For plotting
import matplotlib.dates as mdates   # For formatting dates when plotting
import matplotlib.colors as colors  # For truncating colorbars
import matplotlib.style as style
import xarray as xr                 # For dealing with netCDF data
import pandas as pd                 # A quick way to deal with time stamps
#import netCDF4 as nc                # Another way to deal with netCDF data
import glob
import datetime
import matplotlib.units as munits
import scipy.interpolate
import scipy.signal as sig
import scipy.stats as stats
import sys
import scipy.io as sio

from scipy.optimize import curve_fit
# from netCDF4 import Dataset
from matplotlib import colors as mcolors
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic
from matplotlib.gridspec import GridSpec
from numpy.random import seed
from numpy.random import rand
from scipy.optimize import curve_fit
# matplotlib.use('TKAgg')

print('Done importing modules now')


# --------------- Parameters ---------------------

target_elevation = 60.0  # degrees
path_to_data = "./data_siteH_b1_ALL"
data_save_path = "./data_siteH_VAD/"
figure_save_path = "./plots/VAD_profiles/"


# ---------------- VAD Fitting -------------------


def f(theta, a, b, theta_min):
    # v_r =  a + b * cos(theta - theta_min)
    return a + b * np.cos(theta - theta_min)


def process_file(filename, data_save_path):

    date_time = filename.split(".")
    time = date_time[-2]
    date = date_time[-3]

    dataset = xr.open_dataset(filename, engine='netcdf4')
    # print("\nloaded dataset")
    df = dataset.to_dataframe()
    # print("\nconverted dataset to dataframe")
    # print(len(df.index))
    df_indexes = df.index.to_list()
    ranges = [x[1] for x in df_indexes]
    elevation = df.elevation.to_list()
    # print(len(ranges), len(elevation))
    # Get the measurement altitudes
    measurement_altitudes = []
    for i in range(len(ranges)):
        # get altitude of the measurement
        range_i = ranges[i]
        elevation_i = elevation[i]
        measurement_altitude = range_i*np.cos(np.deg2rad(90-elevation_i))
        measurement_altitudes.append(measurement_altitude)  
    # print("measurement altitudes calculated")
    df["measurement_altitude"] = measurement_altitudes
    # print("measurement altitudes set")

    # only look at df's with elevation = 60 deg and altitude less than 1000 m
    df =  df[df['elevation'] == target_elevation]  
    df = df[df['measurement_altitude'] <= 1000]
    if df.empty:
        print("Dataframe is empty! Skipping... ")
        return

    # get input - output data for each unique altitude:
    unique_altitudes = sorted(df['measurement_altitude'].unique())
    # print("\ngot unique altitudes")

    WS = []
    WD = []
    Altitude = []
    Time = []
    timestamp = df.time_offset.values[0]
    #print(type(timestamp))

    for i, ua in enumerate(unique_altitudes):
        # print(i)
        df_altitude = df[df['measurement_altitude'] == ua]

        radial_velocity = df_altitude["radial_velocity"].values
        azimuth = np.deg2rad(df_altitude["azimuth"].values)
        # initial parameter guess
        p0 = [1, 
              max(max(radial_velocity),1),
              azimuth[np.argmax(radial_velocity)]]
        popt, pcov = curve_fit(f, azimuth, radial_velocity, p0 = p0, bounds=([-np.inf, 0, 0], [np.inf, np.inf, np.inf]))
        predict = [f(theta_i, popt[0], popt[1], popt[2]) for theta_i in azimuth]
        # Calculate R_sq
        corr_matrix = np.corrcoef(radial_velocity, predict)
        corr = corr_matrix[0,1]
        R_sq = corr**2
        a, b, theta_min = popt[0], popt[1], popt[2]

        azimuth_test = np.linspace(0, 2*np.pi, 100)
        predict_test = [f(theta_i, popt[0], popt[1], popt[2]) for theta_i in azimuth_test]

        if date == "20221111" and time == "040020": 
            plt.figure()
            plt.scatter(azimuth, radial_velocity, label="True", color="black", alpha=0.5)
            plt.plot(azimuth_test, predict_test, color="red", label="Prediction")
            textstr = '$R^2={:.2f}$'.format(R_sq)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax = plt.gca()
            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
            plt.ylabel("Radial Velocity [m/s]")
            plt.xlabel("Azimuth Angle [rad]")
            plt.legend(loc=1)
            plt.savefig("./plots/VAD_fits/"+"{}_{}_{}.png".format(date, time, int(ua)), format="png", dpi=500)
            plt.close()

        if R_sq < 0.7:
            #print("R^2 threshold violated, skipping. {}".format(R_sq))
            WS.append(np.nan)
            WD.append(np.nan)
            Altitude.append(ua)
            Time.append(timestamp)
            continue
        else:
            WS.append(b / np.cos(np.deg2rad(target_elevation)))
            WD.append(np.rad2deg(theta_min))
            Altitude.append(ua)
            Time.append(timestamp)

    alex_df = pd.DataFrame(
        list(
        zip(Time,
            WS, 
            WD, 
            Altitude)),
        columns = [
                "time",
                "windspeed", 
                "wind_direction", 
                "altitude"
                ],      
    )
    
    alex_df.to_pickle(data_save_path+"{}_{}.pkl".format(date, time))





def plot_file(filename, save_path):
    file = open(filename,'rb')
    df = pickle.load(file)
    file.close()

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(df["windspeed"], df["altitude"])
    plt.ylabel("Altitude")
    plt.xlabel("Windspeed")
    plt.subplot(1,2,2)
    plt.plot(df["wind_direction"], df["altitude"])
    plt.xlabel("Wind Direction")

    date_time = filename.split(".")[-2]
    pic_name = date_time.split("/")[-1]
    plt.savefig(save_path + pic_name+".png", dpi=500, format="png")
    plt.close()




# ------------- Call Fitting and Plotting Scripts -----------------------
filenames = glob.glob(path_to_data+'/*.cdf')
# filenames = ['~/Documents/CU Boulder/ATOC 5770/project/data_siteH_b1_ALL/arm.lidar.sgp_s6.ppi.b1.20221216.201538.cdf']

for filename in filenames[:]:
    print("Processing: {}".format(filename))
    t0 = time.time()
    process_file(filename,data_save_path)
    print("Done, processed: in {} seconds".format(time.time()-t0))

processed_filenames = glob.glob(data_save_path +'*')
for filename in processed_filenames:
    #print("Plotting: {}".format(filename))
    plot_file(filename, figure_save_path)