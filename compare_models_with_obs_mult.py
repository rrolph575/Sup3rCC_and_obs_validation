

"Compares multiple modelled climate datasets with observed met station data"


import numpy as np
import os
import xarray as xr
from netCDF4 import Dataset
import cftime
import pandas as pd
import matplotlib.pyplot as plt



#%% Load data
def load_station_model_and_obs_data(station_name):
    # %% Load files

    cmip_ifile = os.path.join('Data', f'ds_tas_tasmax_tasmin_{station_name}_ec_earth3_veg.nc') # generated from save_cmip6_data_from_lat_lon.py
    era5_ifile = os.path.join('Data', f'era5_{station_name}_temperature_2m.nc') # generated from save_era5_data_from_lat_lon.py
    obs_ifile = os.path.join('Data', f'temp_daily_max_{station_name}.nc')

    ds_cmip = Dataset(cmip_ifile, mode='r')
    ds_era5 = Dataset(era5_ifile, mode='r')
    ds_obs = Dataset(obs_ifile, mode='r')

    # %% Extract variables
    time_cmip = ds_cmip.variables['time'][:]
    temp_max_cmip = ds_cmip.variables['tasmax'][:]
    temp_max_cmip = temp_max_cmip.filled(np.nan)  # handle masked arrays if any

    time_era5 = ds_era5.variables['time'][:]
    temp_max_era5 = ds_era5.variables['temperature_2m'][:] 
    temp_max_era5 = temp_max_era5.filled(np.nan)  # handle masked arrays if any

    time_obs = ds_obs.variables['time'][:]
    temp_max_obs = ds_obs.variables['temperature_2m'][:] # this is already max temp.  ds_obs.variables['time'].units returns 'days since 2015-01-01 00:00:00' 
    temp_max_obs = temp_max_obs.filled(np.nan) 
    
    #%% Convert to datetimes
    # CMIP (this is daily)
    time_units = ds_cmip.variables['time'].units # returns 'days since 1850-01-01' which is CMIP convention
    time_cmip = time_cmip.filled(np.nan)
    time_cmip_dt = cftime.num2date(time_cmip, units=time_units, only_use_cftime_datetimes=False)

    # ERA5 (this is hourly)
    time_era5 = time_era5.filled(np.nan)
    time_era5_dt = time_era5.astype('datetime64[ns]')
    time_era5_dt = pd.to_datetime(time_era5_dt).to_pydatetime()

    # Obs (this is daily)
    time_obs = ds_obs.variables['time'][:].filled(np.nan) 
    time_units = ds_obs.variables['time'].units
    time_obs_dt = cftime.num2date(time_obs, units=time_units, only_use_cftime_datetimes=False)

    return time_era5_dt, temp_max_era5, time_cmip_dt, temp_max_cmip, time_obs_dt, temp_max_obs




def create_ds_from_time_and_values(time_array, values_array, var_name='max_daily_temperature', units='degC'):
    """
    Combine a time array and corresponding values into an xarray.Dataset.
    
    Parameters
    ----------
    time_array : array-like of datetime objects
        The time coordinates (datetime.datetime or cftime.Datetime objects)
    values_array : array-like
        The values corresponding to each time
    var_name : str
        Name of the variable to store in the dataset
    units : str
        Units of the variable
    
    Returns
    -------
    ds : xarray.Dataset
        Dataset with one variable and a 'time' coordinate
    """
    # Ensure values are numpy array
    values_array = np.array(values_array)
    
    # Ensure time array is pandas datetime index
    time_index = pd.to_datetime(time_array)
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {var_name: (['time'], values_array)},
        coords={'time': time_index}
    )
    ds[var_name].attrs['units'] = units
    
    return ds




#%% Plot one year of max temp data and label how many days exceed threshold in each obs and model data
def plot_daily_max_year_with_threshold(ds_model1, ds_model2, ds_obs, model1_label='', model2_label='', obs_label='', year=None, threshold=30):

    '''
    ## For testing
    ds_model1 = ds_cmip
    model1_label = 'CMIP6 EC-Earth3-Veg'
    ds_model2 = ds_era5
    model2_label = 'ERA5'
    #ds_obs = ds_obs
    obs_label = 'Observed'
    threshold = 30  # degrees C
    year=2015
    '''

    # --- Filter by the year and create data arrays of the variable ---
    da_obs_sel_year = ds_obs['max_daily_temperature'].sel(time=str(year))
    da_model1_sel_year = ds_model1['max_daily_temperature'].sel(time=str(year))
    da_model2_sel_year = ds_model2['max_daily_temperature'].sel(time=str(year))

    # --- Compute day of year for plotting ---
    obs_x = da_obs_sel_year['time'].dt.dayofyear
    model1_x = da_model1_sel_year['time'].dt.dayofyear
    model2_x = da_model2_sel_year['time'].dt.dayofyear

    # --- Count days above threshold ---
    obs_num_days_above = int((da_obs_sel_year > threshold).sum().values)
    model1_num_days_above = int((da_model1_sel_year > threshold).sum().values)
    model2_num_days_above = int((da_model2_sel_year > threshold).sum().values)
    diff_days_obs_minus_model1 = obs_num_days_above - model1_num_days_above  # difference
    diff_days_obs_minus_model2 = obs_num_days_above - model2_num_days_above

    # --- Create month ticks ---
    months = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='MS')
    month_ticks = months.dayofyear
    month_labels = months.month

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12,7))
    plt.rcParams.update({
        'font.size': 18,        # default font size for text
        'axes.titlesize': 16,   # title
        'axes.labelsize': 18,   # x/y labels
        'xtick.labelsize': 12,  # x-tick labels
        'ytick.labelsize': 18,  # y-tick labels
        'legend.fontsize': 12,  # legend
        'figure.titlesize': 18,  # figure title (if using fig.suptitle)
        'xtick.labelsize': 18,
        'ytick.labelsize': 18
    })

    # Shade summer months (JJA)
    jja_start = pd.Timestamp(f'{year}-06-01').dayofyear
    jja_end = pd.Timestamp(f'{year}-08-31').dayofyear
    ax.axvspan(jja_start, jja_end, color='yellow', alpha=0.1, label='JJA')

    # Plot model and observed
    model1_color = 'blue'
    model2_color = 'red'
    obs_color = 'black'
    ax.plot(model1_x, da_model1_sel_year.values, label=model1_label, color=model1_color)
    ax.plot(model2_x, da_model2_sel_year.values, label=model2_label, color=model2_color)
    ax.plot(obs_x, da_obs_sel_year.values, label=obs_label, color=obs_color, linewidth=3)

    # Shade days above threshold
    ax.fill_between(model1_x, da_model1_sel_year.values, threshold, where=(da_model1_sel_year.values>threshold), 
                    color=model1_color, alpha=0.2, label=f'{model1_label} >{threshold}°C')
    ax.fill_between(model2_x, da_model2_sel_year.values, threshold, where=(da_model2_sel_year.values>threshold), 
                    color=model2_color, alpha=0.2, label=f'{model2_label} >{threshold}°C')
    ax.fill_between(obs_x, da_obs_sel_year.values, threshold, where=(da_obs_sel_year.values>threshold), 
                    color=obs_color, alpha=0.05, label=f'{obs_label} >{threshold}°C')

    # Horizontal line at threshold
    ax.axhline(threshold, linestyle='--', linewidth=1, color='k', label=f'{threshold}°C')

    # --- Annotate number of days above threshold in upper-left with background box ---
    bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none')

    '''
    spacing = 0.05
    ax.text(0.01, 0.98, f'Obs days > {threshold}°C: {obs_num_days_above}', 
            transform=ax.transAxes, color=obs_color, verticalalignment='top',
            horizontalalignment='left', bbox=bbox_props)
    ax.text(0.01, 0.98-spacing, f'{model1_label} days > {threshold}°C: {model1_num_days_above}', 
            transform=ax.transAxes, color=model1_color, verticalalignment='top',
            horizontalalignment='left', bbox=bbox_props)
    ax.text(0.01, 0.98-2*spacing, f'{model2_label} days > {threshold}°C: {model2_num_days_above}',
            transform=ax.transAxes, color=model2_color, verticalalignment='top',
            horizontalalignment='left', bbox=bbox_props)
    ax.text(0.01, 0.98-3*spacing, f'{obs_label}- {model1_label} days > {threshold}°C: {diff_days_obs_minus_model1}', 
            transform=ax.transAxes, color='black', verticalalignment='top',
            horizontalalignment='left', bbox=bbox_props)
    ax.text(0.01, 0.98-4*spacing, f'{obs_label}- {model2_label} days > {threshold}°C: {diff_days_obs_minus_model2}', 
            transform=ax.transAxes, color='black', verticalalignment='top',
            horizontalalignment='left', bbox=bbox_props)
    '''


    # --- Labels, ticks, legend ---
    ax.set_title(f"Daily Max Temperature – {year}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    ax.grid(True)
    ax.legend()


    # --- Construct the caption text ---
    caption_text = (
        f'Obs days > {threshold}°C: {obs_num_days_above}\n'
        f'{model1_label} days > {threshold}°C: {model1_num_days_above}\n'
        f'{model2_label} days > {threshold}°C: {model2_num_days_above}\n'
        f'{obs_label} - {model1_label} days > {threshold}°C: {diff_days_obs_minus_model1}\n'
        f'{obs_label} - {model2_label} days > {threshold}°C: {diff_days_obs_minus_model2}'
    )

    # --- Leave space at the bottom of the figure for the caption ---
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # second arg.

    # --- Add the caption below the figure ---
    fig.text(0.03, 0.03, caption_text, ha='left', va='bottom', fontsize=12)


    # --- Save figure ---
    model1_filelabel = model1_label.replace(' ', '_').replace('-', '_')
    model2_filelabel = model2_label.replace(' ', '_').replace('-', '_')
    plt.savefig(f'Figures/daily_max_temp_{year}_{model1_filelabel}_{model2_filelabel}_obs.png')
    plt.show()




#%%
## Compare across all years, not just one year 

def compute_max_by_day_of_year_leapsafe(ds, var_name='max_daily_temperature', common_years=None):
    """
    Compute the maximum value for each day-of-year across all years,
    keeping a consistent 1-366 day axis for leap years.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'time' and the variable to process
    var_name : str
        Name of the variable to compute maxima for
    common_years : list or array, optional
        If provided, only include these years in the calculation
    
    Returns
    -------
    da_max_doy : xarray.DataArray
        Max values for each day-of-year (1..366)
    """
    da = ds[var_name]

    # Select only common years if provided
    if common_years is not None:
        da = da.sel(time=da['time'].dt.year.isin(common_years))

    # Create day-of-year coordinate
    dayofyear = da['time'].dt.dayofyear

    # Group by day-of-year and take max
    da_max_doy = da.groupby(dayofyear).max(dim='time')

    # Reindex to include all 1..366 days
    full_doy = np.arange(1, 367)
    da_max_doy = da_max_doy.reindex(dayofyear=full_doy, fill_value=np.nan)
    
    return da_max_doy


def plot_max_daily_across_years_leapsafe(ds_model1, ds_model2, ds_obs, 
                                         model1_label='', model2_label='', obs_label='', 
                                         threshold=30):
    # --- Find overlapping years ---
    years_model1 = ds_model1['time'].dt.year.values
    years_model2 = ds_model2['time'].dt.year.values
    years_obs    = ds_obs['time'].dt.year.values


    common_years = np.intersect1d(years_model1, years_model2)
    common_years = np.intersect1d(common_years, years_obs)

    if len(common_years) == 0:
        raise ValueError("No overlapping years between datasets!")

    # --- Compute max by day-of-year (leap-safe) ---
    da_model1_max = compute_max_by_day_of_year_leapsafe(ds_model1, common_years=common_years)
    da_model2_max = compute_max_by_day_of_year_leapsafe(ds_model2, common_years=common_years)
    da_obs_max    = compute_max_by_day_of_year_leapsafe(ds_obs,    common_years=common_years)

    # --- Proceed with plotting as before ---
    x = da_obs_max['dayofyear'].values

    # Count days above threshold
    obs_num_days_above = int((da_obs_max > threshold).sum().values)
    model1_num_days_above = int((da_model1_max > threshold).sum().values)
    model2_num_days_above = int((da_model2_max > threshold).sum().values)
    diff_days_obs_minus_model1 = obs_num_days_above - model1_num_days_above
    diff_days_obs_minus_model2 = obs_num_days_above - model2_num_days_above

    # Month ticks (non-leap reference year)
    months = pd.date_range('2001-01-01', '2001-12-31', freq='MS')
    month_ticks = months.dayofyear
    month_labels = months.month

    fig, ax = plt.subplots(figsize=(12,7))
    plt.rcParams.update({
        'font.size': 18,        
        'axes.titlesize': 16,   
        'axes.labelsize': 18,   
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 12,
    })

    # Shade summer months
    jja_start = pd.Timestamp('2001-06-01').dayofyear
    jja_end = pd.Timestamp('2001-08-31').dayofyear
    ax.axvspan(jja_start, jja_end, color='yellow', alpha=0.1, label='JJA')

    # Colors
    model1_color = 'blue'
    model2_color = 'red'
    obs_color = 'black'

    # Plot
    ax.plot(x, da_model1_max.values, label=model1_label, color=model1_color)
    ax.plot(x, da_model2_max.values, label=model2_label, color=model2_color)
    ax.plot(x, da_obs_max.values, label=obs_label, color=obs_color, linewidth=3)

    # Shade above threshold
    ax.fill_between(x, da_model1_max.values, threshold, where=(da_model1_max.values>threshold), 
                    color=model1_color, alpha=0.2)
    ax.fill_between(x, da_model2_max.values, threshold, where=(da_model2_max.values>threshold), 
                    color=model2_color, alpha=0.2)
    ax.fill_between(x, da_obs_max.values, threshold, where=(da_obs_max.values>threshold), 
                    color=obs_color, alpha=0.05)

    ax.axhline(threshold, linestyle='--', linewidth=1, color='k', label=f'{threshold}°C')

    # Labels

    start_year = int(common_years.min())
    end_year   = int(common_years.max())
    year_range_str = f"{start_year}–{end_year}"

    ax.set_title(f"Max Daily Temperature by Day-of-Year from {year_range_str}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    ax.grid(True)
    ax.legend()

    # Caption
    caption_text = (
        f'Obs days > {threshold}°C: {obs_num_days_above}\n'
        f'{model1_label} days > {threshold}°C: {model1_num_days_above}\n'
        f'{model2_label} days > {threshold}°C: {model2_num_days_above}\n'
        f'{obs_label} - {model1_label} days > {threshold}°C: {diff_days_obs_minus_model1}\n'
        f'{obs_label} - {model2_label} days > {threshold}°C: {diff_days_obs_minus_model2}'
    )
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.text(0.03, 0.03, caption_text, ha='left', va='bottom', fontsize=12)

    # Save
    model1_filelabel = model1_label.replace(' ', '_').replace('-', '_')
    model2_filelabel = model2_label.replace(' ', '_').replace('-', '_')
    plt.savefig(f'Figures/max_daily_temp_across_common_years_{model1_filelabel}_{model2_filelabel}_obs.png')
    plt.show()


#%% Procedure

# ---Define which met stations to compare ----
#station_names = ['Mt_Plymouth', 'OCOEE1_4N', 'Orlando4_8NNW', 'Orlando_Intl'] 
station_name = 'Orlando_Intl'


## ------------- Compare one year ----------
year = 2015
threshold = 30 # degrees C
# Load model and obs data ---
time_era5_dt, temp_max_era5, time_cmip_dt, temp_max_cmip, time_obs_dt, temp_max_obs = load_station_model_and_obs_data(station_name)
#Unit conversion ---
#  Convert to C, calc temp max for ERA5, combine the formatted datetimes and data into xarray Datasets
ds_era5 = create_ds_from_time_and_values(time_era5_dt, temp_max_era5, var_name='max_daily_temperature', units='degC')
ds_cmip = create_ds_from_time_and_values(time_cmip_dt, temp_max_cmip, var_name='max_daily_temperature', units='degC')
# Convert CMIP temp (Kelvin) to deg C
ds_cmip['max_daily_temperature'] = ds_cmip['max_daily_temperature'] - 273.15
ds_obs = create_ds_from_time_and_values(time_obs_dt, temp_max_obs, var_name='max_daily_temperature', units='degC')
# Resample ERA5 from hourly to daily, calculate max temp
ds_era5 = ds_era5.resample(time='1D').max()
# Plot ----
plot_daily_max_year_with_threshold(
    ds_cmip,    # ds_model1
    ds_era5,    # ds_model2
    ds_obs,     # ds_obs
    model1_label='CMIP6 EC-Earth3-Veg',
    model2_label='ERA5',
    obs_label='Observed',
    year=year,
    threshold=threshold
)


## ----------- Plot max daily across all years (leap-safe) -------------
plot_max_daily_across_years_leapsafe(
    ds_cmip,
    ds_era5,
    ds_obs,
    model1_label='CMIP6 EC-Earth3-Veg',
    model2_label='ERA5',
    obs_label='Observed',
    threshold=30
)


