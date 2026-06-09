# Libraries
import sys
import os
import glob
import xarray as xr

# Directories
dir_VP = '/glade/derecho/scratch/detouma/extreme_objects/VP/'
dir_anom = '/glade/derecho/scratch/detouma/extreme_objects/VP/mean/'

yy = int(sys.argv[1])
mid_year = yy+1980
print(mid_year, flush=True)

ndays_window = 30
nyears_window = 34

fyear0 = 1920
fyear1 = 2100
year0 = 1980 
year1 = 2100
years = range(year0,(year1+1),1)
nyears = len(years)
ndays = 365

files_list = []
for file in sorted(glob.glob(dir_VP+'*.nc')):
    files_list.append(os.path.relpath(file, dir_VP))

ens_string = [i.split('_')[2] for i in files_list]

window_year0 = int(yy+1980-(nyears_window/2))
window_year1 = int(yy+1980+(nyears_window/2))
if window_year1>year1:
    window_year1=year1


VPD_ens_list = []
print('reading in files', flush=True)
for nn in files_list:
    print(nn, flush=True)
    xr_ens = xr.open_dataset(dir_VP+nn)
    VP_ens = xr_ens.VP.sel(time=xr_ens.time.dt.year.isin(range(window_year0,window_year1+1,1)))
    VPs_ens = xr_ens.VPs.sel(time=xr_ens.time.dt.year.isin(range(window_year0,window_year1+1,1))) 
    VPD_ens = xr.DataArray(VPs_ens-VP_ens, coords=VP_ens.coords, name = 'VPD')
    VPD_ens_list.append(VPD_ens)
print('concat files', flush=True)
VPD = xr.concat(VPD_ens_list, dim='ens_member',coords='minimal')
print('mean across ens', flush=True)
VPD_ens_mean = VPD.mean(dim='ens_member')
print('rechunk', flush=True)
VPD_ens_mean_chunk = VPD_ens_mean.chunk({"time": len(VPD_ens_mean.time), "lat": 10, "lon":10})
print('rolling mean for day window', flush=True)
VPD_ens_mean_rolling = VPD_ens_mean_chunk.rolling(time=ndays_window, center=True).mean()
print('mean for each doy', flush=True)
VPD_ens_mean_rolling_clim = VPD_ens_mean_rolling.groupby('time.dayofyear').mean()
print('writing file', flush=True)
VPD_ens_mean_rolling_clim.to_netcdf(dir_anom+
                                'CESM2-LE_allens_mean_'+str(ndays_window)+
                                'day_mov_'+str(nyears_window)+'year_mov_avg_VPD_'+
                                str(mid_year)+'_NWHemi.nc')

