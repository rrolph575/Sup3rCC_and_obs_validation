import os
import xarray as xr
import sys
import numpy as np



era5_datapath = '/datasets/sup3rwind/era5/conus/processed'

# -----------------------------
# Load multiple ERA5 years
# -----------------------------
def load_era5_years(years,variable='temperature_2m'):
    """
    Lazily loads multiple ERA5 years into a single xarray.Dataset.
    Uses Dask arrays for memory efficiency.
    """
    files = [
        os.path.join(
            era5_datapath,
            str(year),
            f'era5_conus_{year}_{variable}.nc'
        )
        for year in years
    ]

    # Open lazily
    ds = xr.open_mfdataset(
        files,
        combine='by_coords',
        engine='h5netcdf',
        parallel=True,
        data_vars='minimal',
        coords='minimal',
        compat='override'
    )

    return ds


# ---------------------------------------------
# Extract and save a station timeseries
# ---------------------------------------------

def extract_nearest(ds, lat0, lon0, variable='temperature_2m'):
    """
    Select the grid point nearest to (lat0, lon0) from a 2D lat/lon ERA5 dataset.
    """
    # compute squared distance
    dist2 = (ds.latitude - lat0)**2 + (ds.longitude - lon0)**2

    # find index of minimum distance
    # .argmin() flattens the array if no dim is specified
    flat_idx = dist2.argmin().compute()  # compute returns a scalar

    # convert flat index to 2D indices
    idx_south_north, idx_west_east = np.unravel_index(flat_idx, dist2.shape)

    # extract timeseries
    ds_ts = ds[variable].isel(south_north=idx_south_north, west_east=idx_west_east)

    return ds_ts



def save_era5_station_data(ds, station_name, lat0, lon0, variable='temperature_2m'):
    """
    Extracts nearest-point ERA5 data for a station and saves to NetCDF.
    works with Dask, so no need to load full dataset.
    """
    os.makedirs("Data", exist_ok=True)
    filename = os.path.join('Data',f'era5_{station_name}_{variable}.nc')

    if not os.path.exists(filename):
        # Select nearest lat/lon
        ds_station = extract_nearest(ds, lat0, lon0, variable=variable)

        # Save to NetCDF
        ds_station.to_netcdf(filename, engine="netcdf4", format="NETCDF4")
        print(f"File created and saved: {filename}")

        return ds_station

    else:
        # Open existing file lazily
        ds_station = xr.open_dataset(filename, engine='h5netcdf')
        print(f"File exists: {filename}")
        print(f"First 5 times of {variable}:\n", ds_station[variable].isel(time=slice(0, 5)))
        return ds_station
    


# %% Procedure
def main():
    stations = {
        "Mt_Plymouth": (28.799, -81.537),
        "OCOEE1_4N": (28.593, -81.529),
        "Orlando_Intl": (28.418, -81.324),
        "Orlando4_8NNW": (28.572, -81.396)
    }

    # Check command line argument
    if len(sys.argv) < 2: # If you dont add a station name after script name when calling this script from command line then it processes all stations
        print(f"No specific station specified so processing all stations: {list(stations.keys())}")
        stations_to_process = stations

    else:
        station_name = sys.argv[1]  # if you add a station name as an argument to this script name, it only processes that one.
        if station_name not in stations:
            raise ValueError(f"Unknown station '{station_name}'. Valid options: {list(stations.keys())}")
        stations_to_process = {station_name: stations[station_name]}
        print(f"Processing station: {station_name}")


    # Load all years once
    ds_all = load_era5_years(years=range(2015, 2025), variable='temperature_2m')

    # Extract and save for the requested station
    station_data = {}
    for name, (lat0, lon0) in stations_to_process.items():
        ds_station = save_era5_station_data(ds_all, name, lat0, lon0)
        station_data[name] = ds_station

    return station_data


if __name__ == "__main__":
    ds = main()


# Example usage:  python save_era5_data_from_lat_lon.py Mt_Plymouth
    

