
import pandas as pd
import xarray as xr




datapath = 'Data'

# Files contain 2m temp, 2m max and min temp, 2m relative humidity
files = {
    'SE': 'ds_tas_tasmax_tasmin_hurs_Orlando_Intl_ec_earth3_veg.nc', 
    'NW': 'ds_tas_tasmax_tasmin_hurs_Mt_Plymouth_ec_earth3_veg.nc' # This one seems to only has time of observation temp so comparison is not as good to average temp
}

for key, ifile in files.items():
    ds = xr.open_dataset(f'{datapath}/{ifile}', engine='h5netcdf')
    df = ds[['tas','tasmax','hurs']].to_dataframe().reset_index()  # makes time, lat, lon are columns
    # Convert Kelvin to Celsius
    df['tas'] = df['tas'] - 273.15
    df['tasmax'] = df['tasmax'] - 273.15
    # Rename columns to match requested column names
    df = df.rename(columns={
        'time': 'time_index',
        'tas': 'temperature_2m',
        'tasmax': 'max_temp',
        'hurs': 'relativehumidity_2m',
        'lon': 'longitude',
        'lat': 'latitude'
    })
    # Save to csv
    df.to_csv(f'{datapath}/model_input_{key}.csv', index=False)
    
