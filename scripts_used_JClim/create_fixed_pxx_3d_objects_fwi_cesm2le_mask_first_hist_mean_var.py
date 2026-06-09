# Libraries
import numpy as np
from scipy.ndimage import label
import sys
import os
import glob
import xarray as xr

# set nn in bash script
nn = int(sys.argv[1])

#set xx in bassh script
xx = int(sys.argv[2])
# set vv in bash script
vv = int(sys.argv[3])
var_name = ['PRECT','TREFHTMX','RHREFHT','U10'][vv]
print(var_name, flush=True)

n_ens_dist = 100
# Directories
dir_fwi = '/glade/campaign/cgd/cas/detouma/FWI_vars/FWI_hist_mapped/mean_mapped/'
dir_pxx = '/glade/campaign/cgd/cas/detouma/FWI_CESM2LE_pxx/'
dir_event = '/glade/derecho/scratch/detouma/extreme_objects/fwi_objects/NWHemi_3d/'
dir_mask= '/glade/campaign/cgd/cesm/CESM2-LE/lnd/proc/tseries/day_1/SOILWATER_10CM/'

print('reading mask file', flush=True)
mask_xr = xr.open_dataset(dir_mask+'b.e21.BHISTcmip6.f09_g17.LE2-1231.001.clm2.h5.SOILWATER_10CM.18500101-18591231.nc')
mask_xr_sel = mask_xr.sel(lat=slice(0,90), lon=slice(180,360))
lsm = np.array(mask_xr_sel.landmask)
lsm[lsm!=1] = 0

pxx_list = np.array([90,95,98,99,99.9])
pxx = pxx_list[xx]
print('pxx = '+str(pxx),flush=True)

year0 = 1980 #1850
year1 = 2082 
years = range(year0,(year1+1),1)
nyears = len(years)

nyears_window = 35
pxx_year = 1980 

print('reading pxx', flush=True)
f_pxx = 'CESM2-LE_FWI_'+str(pxx_year)+'_'+str(nyears_window)+'-year_moving_high_percentiles_allens_U10_NWHemi.nc'
fwi_pxx_xr = xr.open_dataset(dir_pxx+f_pxx)
fwi_pxx_xr_sel = fwi_pxx_xr.sel(lat=slice(0,90), lon=slice(180,360), quantile=pxx/100)
fwi_pxx_2d = np.array(fwi_pxx_xr_sel.FWI)

print('reading FWI', flush=True)
files_list = [] 
for file in sorted(glob.glob(dir_fwi+'*historic_'+str(pxx_year)+'_mean_mapped*'+var_name+'*.nc')):
 files_list.append(os.path.relpath(file, dir_fwi))

#var_split = [i.split('_')[-2] for i in files_list]
ens_string = [i.split('_')[2] for i in files_list]
print(ens_string[nn], flush=True)

fwi_xr = xr.open_dataset(dir_fwi+files_list[nn])
fwi_xr_sel = fwi_xr.sel(time=fwi_xr.time.dt.year.isin(np.arange(year0,year1,1)))
fwi = np.array(fwi_xr_sel.FWI)

lat = fwi_xr_sel.lat
lon = fwi_xr_sel.lon
time = fwi_xr_sel.time

print('broadcasting fwi pxx', flush=True)
fwi_pxx_3d = np.broadcast_to(fwi_pxx_2d,shape=fwi.shape)

print('fwi binary variable', flush=True)
fwi_bin = np.zeros(fwi.shape,dtype='int')
print('finding exceedances', flush=True)
fwi_bin[fwi>=fwi_pxx_3d] = 1

print('set ocean locations to 0', flush=True)
lsm_3d = np.broadcast_to(lsm,fwi_bin.shape)
fwi_bin[lsm_3d<1] = 0

l_structure = [[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]] # structure for finding connected areas 

print('label() ~ 30 seconds',flush=True)
fwi_labeled, num_labels = label(fwi_bin,l_structure)

print('converting to xarray', flush=True)
xr_event = xr.DataArray(fwi_labeled, name='event', dims=['time','lat','lon'], coords = dict(time=time, lat=lat, lon=lon))
xr_event.attrs['description'] = 'Extreme fire weather events using historically mean mapping for '+var_name+' exceeding the historic p'+str(pxx)+' threshold'
print('writing xarray to netcdf' , flush=True)
xr_event.to_netcdf(dir_event+'CESM2-LE_'+ens_string[nn]+'_mean_mapped_'+var_name+'_to_'+str(n_ens_dist)+'ens_'+str(pxx_year)+'_p'+str(pxx)+'_FWI_event_numbers_'+str(year0)+'-'+str(year1)+'_NWHemi.nc')
