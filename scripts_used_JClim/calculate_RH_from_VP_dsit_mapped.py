import numpy as np
import sys
import glob
import os
import xarray as xr

ens = int(sys.argv[1])
print(ens, flush=True)

dir_T = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/TREFHT/'
dir_RH = '/glade/derecho/scratch/detouma/extreme_objects/RH_from_VP_mapped/dist_mapped/'
dir_VP = '/glade/derecho/scratch/detouma/extreme_objects/VP/dist_mapped/'

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

VP_file = 'CESM2-LE_'+T_ens_string+'_VP_mapped_to_1980_70_ens_distribution_1980-2082.nc'
xr_VP = xr.open_dataset(dir_VP+VP_file)
VP = xr_VP['VP']

VPs = 6.11 * 10**((7.5*T)/(T+237.3))
RH = VP/VPs * 100

RH_xr = xr.Dataset(
    data_vars={
        'RHREFHT': (('time', 'lat', 'lon'), RH.data, {'units': '%', 'longname': 'Relative Humidity'})
    },
    coords={
        'time': RH.time,
        'lat': RH.lat,
        'lon': RH.lon,
    },
    attrs={'title': 'RHREFHT calcuated using distribution mapped VP for CESM2LE_' + T_ens_string}
)

RH_xr = RH_xr.drop_vars("ens_member")

print('writing file', flush=True)
RH_xr.to_netcdf(dir_RH+'CESM2LE_RHREFHT_using_VP_dist_mapped_'+T_ens_string+'_'+str(year0)+'-'+str(year1)+'_NWHemi.nc')
