

import os
import xarray as xr
xr.set_options(use_new_combine_kwarg_defaults=True) # Already implements new xarray dev to avoid warnings when opening multiple files 
import sys



## Define paths
cmip6_datapath = '/datasets/cmip6/ec_earth3_veg/'


## To open multiple files
def load_cmip6_dataset():
    # Open files 
    # Example files (2015 through 2060 available)
    # tas_day_EC-Earth3-Veg_ssp245_r1i1p1f1_gr_20150101-20151231.nc
    # tasmax_day_EC-Earth3-Veg_ssp245_r1i1p1f1_gr_20150101-20151231.nc
    # tasmin_day_EC-Earth3-Veg_ssp245_r1i1p1f1_gr_20150101-20151231.nc
    files_tas = os.path.join(cmip6_datapath, 'tas_day_EC-Earth3-Veg_ssp245_r1i1p1f1_gr_20*.nc')
    files_tasmax = os.path.join(cmip6_datapath, 'tasmax_day_EC-Earth3-Veg_ssp245_r1i1p1f1_gr_20*.nc')
    files_tasmin = os.path.join(cmip6_datapath, 'tasmin_day_EC-Earth3-Veg_ssp245_r1i1p1f1_gr_20*.nc')

    open_kwargs = dict(
    combine='by_coords',
    engine='h5netcdf',
    parallel=False,
    chunks={'time': 365},
    data_vars='minimal',
    coords='minimal',
    compat='override'
    )

    ds_tas = xr.open_mfdataset(files_tas, **open_kwargs)
    ds_tasmax = xr.open_mfdataset(files_tasmax, **open_kwargs)
    ds_tasmin = xr.open_mfdataset(files_tasmin, **open_kwargs)

    ds = xr.merge([ds_tas, ds_tasmax, ds_tasmin])  

    return ds

def save_cmip6_data_from_lat_lon(ds, station_name, lat0, lon0):
    """Extracts and saves CMIP6 data for a given latitude and longitude.

    Args:
        lat0 (float): Latitude of the target location.
        lon0 (float): Longitude of the target location.
    """
    filename = f'ds_tas_tasmax_tasmin_{station_name}_ec_earth3_veg.nc'
    if not os.path.exists(f'Data/{filename}'):
            
            # Convert station lon to 0–360 because our station lat/lon are in -180 to 180 degree format and cmip is in 0-360 format
            lon0 = lon0 % 360

            # compute distance to each point (lazy Dask array)
            #dist = ((ds.lat - lat0)**2 + (ds.lon - lon0)**2)

            # get index of the minimum distance
            #target_idx = dist.argmin(dim='lat').argmin(dim='lon').compute()

            # select timeseries data with the target index
            #ds_ts = ds.isel(lat=target_idx['lat'], lon=target_idx['lon'])
                
            ds_ts = ds.sel(
            lat=lat0,
            lon=lon0,
            method="nearest"
            )


            # Save to NetCDF
            ds_ts.to_netcdf(f"Data/{filename}", engine="netcdf4", format="NETCDF4")
            print(f'File was created and saved to Data/{filename}')

            return ds_ts

    else:
        # Read from file 
        ds_ts = xr.open_dataset(f"Data/{filename}")
        print(
            "File exists. First 5 days of tasmax:\n",
            ds_ts["tasmax"].isel(time=slice(0, 5))
        )
        return ds_ts


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

    ds_all = load_cmip6_dataset()

    ds = save_cmip6_data_from_lat_lon(
        ds_all, station_name, lat0, lon0
    )

    return ds


if __name__ == "__main__":
    ds = main()


# Call script like:  python save_cmip6_data_from_lat_lon.py Mt_Plymouth