
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from rex import MultiFileResourceX, ResourceX
import xarray as xr
from matplotlib.patches import Patch

### Define paths
supercc_datapath = '/datasets/sup3rcc/conus_ecearth3veg_ssp245_r1i1p1f1/v0.2.2/' 


## To open multiple files
def load_supercc_dataset():
    # Open files (2015 through 2059 available)
    files_super = os.path.join(supercc_datapath, 'sup3rcc_conus_ecearth3veg_ssp245_r1i1p1f1_trh_20*.h5')
    ds_super_all = xr.open_mfdataset(files_super, engine='rex', compat='override', coords='minimal')
    return ds_super_all



def save_supercc_data_from_lat_lon(ds_super_all, station_name, lat0, lon0):
    """Extracts and saves SuperCC data for a given latitude and longitude.

    Args:
        lat0 (float): Latitude of the target location.
        lon0 (float): Longitude of the target location.
    """
    filename = f'temp_daily_max_{station_name}.nc'

    # compute distance to each point (lazy Dask array)
    dist = ((ds_super_all.latitude - lat0)**2 + (ds_super_all.longitude - lon0)**2)

    # get index of the minimum distance
    target_gid = dist.argmin(dim='gid').compute()

    # select timeseries data with the target_gid
    ts = ds_super_all.sel(gid=target_gid)
    rh_ts = ts['relativehumidity_2m']     # RH time series
    temp_ts = ts['temperature_2m']        # Temp time series
    df = ts[['relativehumidity_2m', 'temperature_2m']].to_dataframe()

    ### Calculate the max daily temp for the model data and save to NetCDF (if not already saved) 
    if not os.path.exists("Data/{filename}"):
        ## Calculate max daily temp of all years from files_super
        temp_daily_max = ts['temperature_2m'].resample(time='1D').max()
        temp_daily_max_float = temp_daily_max.compute().astype("float32") # Cannot save dtype int16 to netcdf

        ## Save to NetCDF
        ds = xr.Dataset({"temperature_2m": temp_daily_max_float})
        os.makedirs("Data", exist_ok=True)
        # Prepare data
        data = temp_daily_max.values.astype("float32")
        time = temp_daily_max["time"].values
        # Handle scalar coordinates robustly
        attrs = {}
        for k in temp_daily_max.coords:
            coord = temp_daily_max.coords[k]
            if coord.shape == ():  # scalar
                val = coord.values
                # Decode bytes if needed
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                attrs[k] = val
        # Create clean Dataset
        ds_clean = xr.Dataset(
            {"temperature_2m": ("time", data)},
            coords={"time": time},
            attrs=attrs
        )

        # Save to NetCDF
        ds_clean.to_netcdf(f"Data/{filename}", engine="netcdf4", format="NETCDF4")
        return ds_clean
        print(f'File was created and saved to Data/{filename}')
    else:
        # Read from file 
        ds = xr.open_dataset(f"Data/{filename}")
        temp_daily_max = ds['temperature_2m']
        print(f'File exists and here are the first lines of max 2m temp data:\n{temp_daily_max.isel(time=slice(0,5))}')

        return ds 


# %% Procedure
import sys

# %% Procedure
def main():
    stations = {
        "Mt_Plymouth": (28.799, -81.537),
        "OCOEE1_4N": (28.593, -81.529),
        "Orlando_Intl": (28.418, -81.324),
        "Orlando4_8NNW": (28.572, -81.396)
    }

    if len(sys.argv) < 2:
        raise ValueError(
            f"Please provide a station name. Options: {list(stations.keys())}"
        )

    station_name = sys.argv[1]

    if station_name not in stations:
        raise ValueError(
            f"Unknown station '{station_name}'. "
            f"Valid options: {list(stations.keys())}"
        )

    lat0, lon0 = stations[station_name]

    ds_super_all = load_supercc_dataset()

    ds = save_supercc_data_from_lat_lon(
        ds_super_all, station_name, lat0, lon0
    )

    return ds

if __name__ == "__main__":
    main()


# Call script like:  python save_supercc_data_from_lat_lon.py Mt_Plymouth
