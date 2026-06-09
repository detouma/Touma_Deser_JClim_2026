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
vv = int(sys.argv[2])

year0 = 1980
year1 = 2082
years = np.arange(year0,year1+1,1)
nyears = len(years)

var_abbr = ['tmax','pr','rh','wind'][vv]
var_name = ['TREFHTMX','PRECT','RHREFHT','U10'][vv]

print('var = '+ var_name, end = ' | ', flush=True)
print('ens = '+str(ens), flush=True)

# directories
dir_mapped = '/glade/derecho/scratch/detouma/extreme_objects/'+var_name+'/'
dir_var = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/'+var_name+'/'
dir_mean = '/glade/campaign/cgd/cas/detouma/FWI_vars/'+var_name+'/'

# function to preprocess variable datasets
def xr_coord_sel(ds):
 return ds.sel(lon=slice(180,360), lat=slice(0,90))

#read in precipitation
print('read in var', flush=True)
var_files = []
for ff in sorted(glob.glob(dir_var+'b.e21.B*')):
 var_files.append(os.path.relpath(ff, dir_var))

var_files_ens_str0 = [i.split('-', 1)[1].split('.')[0] for i in var_files]
var_files_ens_str1 = [i.split('-', 1)[1].split('.')[1] for i in var_files]
var_files_ens_str = list(map('.'.join, zip(var_files_ens_str0, var_files_ens_str1)))

ens_unique = sorted(list(set(var_files_ens_str)))
ens_string = ens_unique[ens]
print(ens_string, flush=True)

var_files_ens_inds = np.where(np.array(var_files_ens_str)==ens_string)[0]

var_files_ens = []

for ii in var_files_ens_inds:
 var_files_ens.append(var_files[ii])

xr_var = xr.open_mfdataset([dir_var+x for x in var_files_ens], preprocess=xr_coord_sel)
xr_var_time0 = xr_var.sel(time=xr_var.time.dt.year.isin(range(year0,year1+1,1)))
xr_var_time = xr_var_time0[var_name].sel(time=~xr_var_time0.get_index("time").duplicated()) 

var_all = np.array(xr_var_time)

#preprocessing function for reading multiyear files
def preprocess(ds):
    return ds.expand_dims(year = [datetime.now()])

#calculate var anomaly
print('calculate var anomaly', flush=True)
var_mov_mean_files_list = []
for ff in sorted(glob.glob(dir_mean+'CESM2-LE_allens_mean_30day_mov_34year_mov_avg_'+var_name+'*')):
 var_mov_mean_files_list.append(ff)

var_mov_mean_xr = xr.open_mfdataset(var_mov_mean_files_list, preprocess=preprocess)
var_mov_mean_xr['year'] =  years
var_mov_mean_doy = var_mov_mean_xr[var_name].sel(lon=slice(180,360), lat=slice(0,90))
var_mov_mean = np.reshape(np.array(var_mov_mean_doy),newshape=((nyears*365),len(var_mov_mean_doy.lat),len(var_mov_mean_doy.lon)))

var_hist_mean_365 = np.array(var_mov_mean_xr[var_name].sel(year=1980,lon=slice(180,360), lat=slice(0,90)))
var_hist_mean = np.tile(var_hist_mean_365,[nyears,1,1])

if (var_name == 'TREFHTMX'):
	var_mov_anom = var_all - var_mov_mean
	var_hist_mean_mapped = var_mov_anom + var_hist_mean 
else:
	var_mov_anom = var_all/var_mov_mean
	var_hist_mean_mapped = var_mov_anom * var_hist_mean	

print('converting np final array to xr array', flush=True)
var_hist_mean_mapped_xr = xr.DataArray(var_hist_mean_mapped, name=var_name, dims=["time", "lat", "lon"], coords=dict(time=xr_var_time.time, lat=xr_var.lat, lon=xr_var.lon,), attrs=dict(description=var_name+ ' mapped to 1980 mean', units=xr_var_time.units),)

print('writing final xr array file', flush=True)
write_start = time.time()
var_hist_mean_mapped_xr.to_netcdf(dir_mapped+'CESM2-LE_'+ens_string+'_'+str(year0)+'-'+str(year1)+'_members_'+var_name+'_mapped_to_1980_mean.nc')
write_end = time.time()
write_time = write_end - write_start
print('time for writing = '+str(write_time)+'s', flush=True)
