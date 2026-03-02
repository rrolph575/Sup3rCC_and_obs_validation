
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from rex import MultiFileResourceX, ResourceX
import xarray as xr
from matplotlib.patches import Patch

### Define paths
datapath = 'Data/'


## Select if you want to look at southeast (SE) or northwest (NW) region of Orlando
loc = 'SE' # for now only 'SE' available to run because NW seems to only have time of observation temp and not avearge temp. Can add NW obs data here if available.

### Read met station data (1 Jan 2020-1 Jan 2025)
if loc == 'SE':
    obs_df = pd.read_csv(datapath + 'GHCNd_met_station_orlando_intl.csv') 
elif loc == 'NW':
    obs_df = pd.read_csv(datapath)

### Read CMIP model data
files = {
    'SE': 'ds_tas_tasmax_tasmin_hurs_Orlando_Intl_ec_earth3_veg.nc',
    'NW': 'ds_tas_tasmax_tasmin_hurs_Mt_Plymouth_ec_earth3_veg.nc'
}


cmip_ds = xr.open_dataset(f'{datapath}/{files[loc]}', engine='h5netcdf')
# Extract 2m temp to compare
temp_daily = cmip_ds['tas'] - 273.15 # convert from K to C


### Met station data (from https://www.ncei.noaa.gov/cdo-web/search) 
# AWND - Average daily wind speed (mph)
# PGTM - Peak gust time (HHMM)
# TAVG - Average daily temperature (C)
# TMAX - Maximum daily temperature (C)
# TMIN - Minimum daily temperature (C)
# WDF2 - Direction of fastest 2-minute wind speed (degrees)
# WDF5 - Direction of fastest 5-second wind speed (degrees)
# WSF2 - Fastest 2-minute wind speed (mph)
# WSF5 - Fastest 5-second wind speed (mph)
# WT01 - Fog, ice fog, or freezing fog (may include heavy fog)
# WT02 - Heavy fog or heaving freezing fog (not always distinguished from fog)
# WT03 - Thunder
# WT08 - Smoke or haze



### CMIP6 data
# tas - Near surface air temperature (C)
# hurs - Relative humidity at surface (%)




def plot_multiyear_window_climatology(temp_daily, obs_df, 
                                          start_year=2019, end_year=2024,
                                          threshold=30, temp_metric='Avg'):
    """
    For each day-of-year (1..365), compute the maximum temperature across all
    years in the window for both observations and model. Produce a 365-day plot,
    shade days above threshold, and display the difference in number of threshold exceedances.
    """

    # -------------------------
    # OBSERVATIONS
    # -------------------------
    obs_df = obs_df.copy()
    obs_df["DATE"] = pd.to_datetime(obs_df["DATE"])
    mask = (obs_df["DATE"].dt.year >= start_year) & (obs_df["DATE"].dt.year <= end_year)
    obs_win = obs_df[mask].set_index("DATE")
    if temp_metric == 'max_temp':
        obs_tmax_c = (obs_win["TMAX"] - 32) * 5/9 # convert F to C
        obs_by_doy = obs_tmax_c.groupby(obs_tmax_c.index.dayofyear).max()
        obs_curve = obs_by_doy.reindex(range(1, 366))
        temp_metric_label = "Max"
    else:
        temp_metric_label = "Average"
        obs_t_c = (obs_win["TAVG"] - 32) * 5/9 # convert F to C
        obs_by_doy = obs_t_c.groupby(obs_t_c.index.dayofyear)
        obs_curve = obs_by_doy.mean().reindex(range(1, 366))

    # -------------------------
    # MODEL
    # -------------------------
    model_win = temp_daily.sel(time=slice(str(start_year), str(end_year)))
    model_series = model_win.to_pandas().squeeze()
    model_by_doy = model_series.groupby(model_series.index.dayofyear).max()
    model_curve = model_by_doy.reindex(range(1, 366))

    # -------------------------
    # PLOTTING
    # -------------------------
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(obs_curve.index, obs_curve.values, label=f"Observed {temp_metric_label} Across Window", color="blue")
    ax.plot(model_curve.index, model_curve.values, label=f"Model {temp_metric_label} Across Window", color="red")
    
    # Threshold line
    ax.axhline(threshold, linestyle="--", color="k", linewidth=1)

    # Shade days above threshold
    obs_above = obs_curve > threshold
    model_above = model_curve > threshold
    ax.fill_between(obs_curve.index, threshold, obs_curve.values,
                    where=obs_above, color="blue", alpha=0.2, label="Obs > Threshold")
    ax.fill_between(model_curve.index, threshold, model_curve.values,
                    where=model_above, color="red", alpha=0.2, label="Model > Threshold")

    # Month ticks
    months = pd.date_range("2001-01-01", "2001-12-31", freq="MS")  # dummy non-leap year
    month_ticks = months.dayofyear
    month_labels = months.month
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)

    ax.set_xlabel("Month")
    ax.set_ylabel(f"{temp_metric_label} Temperature Across Window (°C)")
    if temp_metric == 'max_temp':
        ax.set_title(f"Multi-Year Maximum Daily Max Temperature Climatology ({start_year}–{end_year})")
    else: 
        ax.set_title(f"Multi-Year Daily Average Temperature Climatology ({start_year}–{end_year})")
    ax.grid(True)
    ax.legend()

    # -------------------------
    # DAYS ABOVE THRESHOLD
    # -------------------------
    obs_days = obs_above.sum()
    model_days = model_above.sum()
    diff_days = obs_days - model_days  # obs minus model

    # Add difference to upper left corner
    ax.text(0.02, 0.95, f"Obs - Model days above {threshold}°C: {diff_days}",
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'Figures/multiyear_window_{temp_metric_label}_climatology_{start_year}_{end_year}.png')
    #plt.show()

    return obs_curve, model_curve, obs_days, model_days, diff_days



###% Procedure
plot_multiyear_window_climatology(temp_daily, obs_df,
                                          start_year=2015, end_year=2024,
                                          threshold=30, temp_metric='Avg')  

# Notes on function:
# set temp_metric = max_temp if you want to compare daily maxiumum temp 
# threshold is temperature threshold for comparison in degrees celsius
