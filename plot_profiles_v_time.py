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
import matplotlib.pylab as pl
import xarray as xr                 # For dealing with netCDF data
import pandas as pd                 # A quick way to deal with time stamps
#import netCDF4 as nc                # Another way to deal with netCDF data


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

# --------------- Parameters ---------------------

target_elevation = 60.0  # degrees

data_path = "./data_siteH_VAD/"
data_path = "./data_siteH_VAD/"
figure_save_path = "./plots/"




def plot_file(jet_set, save_path, flip_wd = False):
    dates = []
    altitudes = []
    windspeeds = []
    wind_directions = []
    for i, filename in enumerate(jet_set):
        filename = data_path+filename
        file = open(filename,'rb')
        df = pickle.load(file)
        file.close()
        dates.append(df["time"][0])
        altitudes.append(df["altitude"])
        windspeeds.append(df["windspeed"])
        if flip_wd:
            wind_directions.append((df["wind_direction"]+360)%360)
        else:
            wind_directions.append(df["wind_direction"])
    date_time = filename.split(".")[-2]
    pic_name = date_time.split("/")[-1]
    time_deltas = [(x - min(dates))/pd.offsets.Minute(1) for x in dates]


    plt.figure()
    fig, ax = plt.subplots()
    lc = multiline(windspeeds, altitudes, time_deltas, cmap='viridis_r', lw=2)
    axcb = fig.colorbar(lc)
    axcb.set_label('Time from LLJ Formation [min]')
    ax.set_xlabel("Windspeed [m/s]")
    ax.set_ylabel("Altitude [m]")
    plt.tight_layout()
    plt.savefig(save_path + pic_name+"_WS.png", dpi=500, format="png")
    plt.close()


    plt.figure()
    fig, ax = plt.subplots()
    lc = multiline(wind_directions, altitudes, time_deltas, cmap='viridis_r', lw=2)
    axcb = fig.colorbar(lc)
    axcb.set_label('Time from LLJ Formation [min]')
    ax.set_xlabel("Wind Direction [deg]")
    ax.set_ylabel("Altitude [m]")
    plt.tight_layout()
    plt.savefig(save_path + pic_name+"_WD.png", dpi=500, format="png")
    plt.close()



# ------------- Call Fitting and Plotting Scripts ----------------------

nov_10_set = [
    "20221110_220020.pkl",
    "20221110_223002.pkl",
    "20221110_230020.pkl",
    "20221110_233003.pkl",
    "20221111_000020.pkl",
    "20221111_003003.pkl",
    "20221111_010020.pkl",
    "20221111_013003.pkl",
    "20221111_020020.pkl",
    "20221111_023003.pkl",
    "20221111_030020.pkl"
]
nov_13_set = [
    "20221113_073003.pkl",
    "20221113_080020.pkl",
    "20221113_090021.pkl",
    "20221113_093003.pkl",
    "20221113_120020.pkl"
]
nov_14_set = [
    "20221114_130020.pkl",
    "20221114_133002.pkl",
    "20221114_140020.pkl"
]


set_list = [nov_10_set, nov_13_set, nov_14_set]


print("\nPlotting profiles...")
t0 = time.time()
for i, jet_set in enumerate(set_list):
    print("Plotting: {}".format(jet_set))
    plot_file(jet_set, figure_save_path, flip_wd=(i>0))