import numpy as np
import sys
import glob
import os
import xarray as xr

ens = int(sys.argv[1])
print(ens, flush=True)

dir_T = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/TREFHT/'
dir_RH = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/RHREFHT/'
dir_VP = '/glade/derecho/scratch/detouma/extreme_objects/VP/'

year0 = 1920
year1 = 2100

T_files_list = []
for file in sorted(glob.glob(dir_T+'*.nc')):
 T_files_list.append(os.path.relpath(file, dir_T))

T_ens_split = [i.split('-',1)[1][0:8] for i in T_files_list]
T_ens_unique = sorted(list(set(T_ens_split)))
T_ens_string = T_ens_unique[ens]
T_files_inds = np.where(np.array(T_ens_split)==T_ens_string)[0]
T_files_ens = list(dir_T+T_files_list[i] for i in list(T_files_inds))

xr_T = xr.open_mfdataset(T_files_ens)

xr_T_sel0 = xr_T.sel(time=xr_T.time.dt.year.isin(range(year0,year1+1,1)), lat=slice(0,90), lon=slice(180,360))
xr_T_sel = xr_T_sel0.sel(time=~xr_T_sel0.get_index("time").duplicated())

T = xr_T_sel['TREFHT']-273.15

RH_files_list = []
for file in sorted(glob.glob(dir_RH+'*.nc')):
 RH_files_list.append(os.path.relpath(file, dir_RH))

RH_ens_split = [i.split('-',1)[1][0:8] for i in RH_files_list]
RH_ens_unique = sorted(list(set(RH_ens_split)))
RH_ens_string = RH_ens_unique[ens]
RH_files_inds = np.where(np.array(RH_ens_split)==RH_ens_string)[0]
RH_files_ens = list(dir_RH+RH_files_list[i] for i in list(RH_files_inds))

xr_RH = xr.open_mfdataset(RH_files_ens)

xr_RH_sel0 = xr_RH.sel(time=xr_RH.time.dt.year.isin(range(year0,year1+1,1)), lat=slice(0,90), lon=slice(180,360))
xr_RH_sel = xr_RH_sel0.sel(time=~xr_RH_sel0.get_index("time").duplicated())

RH = xr_RH_sel['RHREFHT']
# RH[RH>100] = 100

VPs = 6.11 * 10**((7.5*T)/(T+237.3))
VP = RH/100 * VPs

VP_xr = xr.Dataset(
    data_vars={
        'VPs': (('time', 'lat', 'lon'), VPs.data, {'units': 'hPA', 'longname': 'Saturation Vapor Pressure'}),
        'VP': (('time', 'lat', 'lon'), VP.data, {'units': 'hPa', 'longname': 'Vapor Pressure'})
    },
    coords={
        'time': VP.time,
        'lat': VP.lat,
        'lon': VP.lon,
    },
    attrs={'title': 'VPs and VP calculated using RHREFHT and TREFHT for CESM2LE_' + T_ens_string}
)

print('writing file', flush=True)
VP_xr.to_netcdf(dir_VP+'CESM2LE_VapPress_'+T_ens_string+'_'+str(year0)+'-'+str(year1)+'_NWHemi.nc')
