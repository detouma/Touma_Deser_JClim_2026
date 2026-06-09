import numpy as np
import sys
import os
import glob
import xarray as xr
import time
from datetime import datetime

script_start = time.time()
# argument inputs
ens = int(sys.argv[1])


year0 = 1980
year1 = 2083
years = np.arange(year0,year1+1,1)
nyears = len(years)

var_abbr = 'VP' 
var_name = 'VapPress'

print('var = '+ var_name, end = ' | ', flush=True)
print('ens = '+str(ens), flush=True)

# directories
dir_mapped = '/glade/derecho/scratch/detouma/extreme_objects/VP/'
dir_var = '/glade/campaign/cgd/cas/detouma/FWI_vars/VP/'
dir_mean = '/glade/derecho/scratch/detouma/extreme_objects/VP/mean/'


# finding files for variable and nyear window
files_list = []
for file in sorted(glob.glob(dir_var+'*.nc')):
	files_list.append(os.path.relpath(file, dir_var))

ens_string = [i.split('_')[2] for i in files_list]

xr_ens0 = xr.open_dataset(dir_var+files_list[0])
lat0 = xr_ens0.lat
lon0 = xr_ens0.lon
lat_inds = np.where(lat0>=0)[0]
lon_inds = np.where(lon0>=180)[0]
lat = lat0[lat_inds]
lon = lon0[lon_inds]
nlat = len(lat)
nlon = len(lon)

xr_ens0_units = xr_ens0[var_abbr].units 

var_file = files_list[ens]

xr_var = xr.open_dataset(dir_var+var_file)
xr_var_time0 = xr_var.sel(time=xr_var.time.dt.year.isin(range(year0,year1+1,1)))
xr_var_time = xr_var_time0[var_abbr].sel(time=~xr_var_time0.get_index("time").duplicated()) 

var_all = np.array(xr_var_time)

#preprocessing function for reading multiyear files
def preprocess(ds):
    return ds.expand_dims(year = [datetime.now()])

#calculate var anomaly
print('calculate var anomaly', flush=True)
var_mov_mean_files_list = []
for ff in sorted(glob.glob(dir_mean+'CESM2-LE_allens_mean_30day_mov_34year_mov_avg_'+var_name+'_*')):
 var_mov_mean_files_list.append(ff)

var_mov_mean_xr = xr.open_mfdataset(var_mov_mean_files_list, preprocess=preprocess)
var_mov_mean_xr['year'] =  years
var_mov_mean_doy = var_mov_mean_xr[var_abbr].sel(lon=slice(180,360), lat=slice(0,90))
var_mov_mean = np.reshape(np.array(var_mov_mean_doy),newshape=((nyears*365),len(var_mov_mean_doy.lat),len(var_mov_mean_doy.lon)))

var_hist_mean_365 = np.array(var_mov_mean_xr[var_abbr].sel(year=1980,lon=slice(180,360), lat=slice(0,90)))
var_hist_mean = np.tile(var_hist_mean_365,[nyears,1,1])

var_mov_anom = var_all/var_mov_mean
var_hist_mean_mapped = var_mov_anom * var_hist_mean	

print('converting np final array to xr array', flush=True)
var_hist_mean_mapped_xr = xr.DataArray(var_hist_mean_mapped, name=var_abbr, dims=["time", "lat", "lon"], coords=dict(time=xr_var_time.time, lat=xr_var.lat, lon=xr_var.lon,), attrs=dict(description=var_name+ ' mapped to 1980 mean', units=xr_var_time.units),)

print('writing final xr array file', flush=True)
write_start = time.time()
var_hist_mean_mapped_xr.to_netcdf(dir_mapped+'CESM2-LE_'+ens_string[ens]+'_'+str(year0)+'-'+str(year1)+'_members_'+var_abbr+'_mapped_to_1980_mean.nc')
write_end = time.time()
write_time = write_end - write_start
print('time for writing = '+str(write_time)+'s', flush=True)
