# Libraries
import numpy as np
import sys
import os
import glob
import xarray as xr

# Directories
dir_wind = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/U10/'
dir_anom = '/glade/derecho/scratch/detouma/extreme_objects/U10/'

yy = int(sys.argv[1])
mid_year = yy+1980
print(mid_year, flush=True)

ndays_window = 30
nyears_window = 34

year0 = 1980 
year1 = 2100
years = range(year0,(year1+1),1)
nyears = len(years)
ndays = 365

files_list = []
for file in sorted(glob.glob(dir_wind+'*.nc')):
    files_list.append(os.path.relpath(file, dir_wind))

ens_split = [i.split('-', 1)[1] for i in files_list]
ens_bb = [i.split('.')[2] for i in files_list]
ens_string0 = [i.split('.')[0] for i in ens_split]
ens_string1 = [i.split('.')[1] for i in ens_split]
ens_string = list(map('.'.join, zip(ens_string0, ens_string1)))

fdate_all = [i.split('.')[9] for i in files_list]
fdate0 = np.array([i.split('-')[0] for i in fdate_all]).astype(int)
fdate1 = np.array([i.split('-')[1] for i in fdate_all]).astype(int)
fyyyy0 = np.floor(fdate0/10000).astype(int)
fyyyy1 = np.floor(fdate1/10000).astype(int)

window_year0 = int(yy+1980-(nyears_window/2))
window_year1 = int(yy+1980+(nyears_window/2))
if window_year1>year1:
    window_year1=year1

fdate_inds = np.where((fyyyy0<=window_year1)&(fyyyy1>=window_year0))[0]
files_window = list(files_list[i] for i in list(fdate_inds))
ens_window = list(ens_string[i] for i in list(fdate_inds))
ens_unique = list(set(ens_window))
print(files_window)

wind_ens_list = []
print('reading in files', flush=True)
for nn in ens_unique:
    print(nn, end=', ', flush=True)
    ens_inds = np.where(np.array(ens_window)==nn)[0]
    files_ens = list(files_window[i] for i in list(ens_inds))
    xr_ens = xr.open_mfdataset([dir_wind+x for x in files_ens],
                            )
    wind_ens_list.append(xr_ens.U10.sel(lat=slice(0,90), lon=slice(180,360)))
print('concat files', flush=True)
wind_all = xr.concat(wind_ens_list, dim='ens_member',coords='minimal')
print('time selection', flush=True)
wind = wind_all.sel(time=wind_all.time.dt.year.isin(range(window_year0,window_year1+1,1)))
print('mean across ens', flush=True)
wind_ens_mean = wind.mean(dim='ens_member')
print('rechunk', flush=True)
wind_ens_mean_chunk = wind_ens_mean.chunk({"time": len(wind_ens_mean.time), "lat": 10, "lon":10})
print('rolling mean for day window', flush=True)
wind_ens_mean_rolling = wind_ens_mean_chunk.rolling(time=ndays_window, center=True).mean()
print('mean for each doy', flush=True)
wind_ens_mean_rolling_clim = wind_ens_mean_rolling.groupby('time.dayofyear').mean()
print('writing file', flush=True)
wind_ens_mean_rolling_clim.to_netcdf(dir_anom+
                                'CESM2-LE_allens_mean_'+str(ndays_window)+
                                'day_mov_'+str(nyears_window)+'year_mov_avg_U10_'+
                                str(mid_year)+'_NWHemi.nc')

