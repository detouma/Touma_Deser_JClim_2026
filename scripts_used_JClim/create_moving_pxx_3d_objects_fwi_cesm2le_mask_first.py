# Libraries
import numpy as np
from netCDF4 import Dataset 
from scipy.ndimage.measurements import label
import sys
import os
import glob

# Directories
dir_fwi = '/glade/campaign/cgd/cas/detouma/FWI_CESM2LE/'
# dir_pxx = '/glade/campaign/cgd/cas/detouma/FWI_CESM2LE_pxx/'
dir_pxx = '/glade/derecho/scratch/detouma/fire-precip/FWI/FWI_pxx_CESM2LE/'
dir_event = '/glade/derecho/scratch/detouma/extreme_objects/fwi_objects/'
dir_mask= '/glade/campaign/cgd/cesm/CESM2-LE/lnd/proc/tseries/day_1/SOILWATER_10CM/'

nc_mask = Dataset(dir_mask+'b.e21.BHISTcmip6.f09_g17.LE2-1231.001.clm2.h5.SOILWATER_10CM.18500101-18591231.nc','r')
lat_mask = nc_mask.variables['lat'][:]
lon_mask = nc_mask.variables['lon'][:]
lat_mask_inds = np.where(lat_mask>=0)[0]
lon_mask_inds = np.where(lon_mask>=180)[0]
lsm = nc_mask.variables['landmask'][lat_mask_inds,lon_mask_inds]
lsm[lsm!=1] = 0

# set nn in bash script
nn = int(sys.argv[1])

pxx_list = np.array([90,95,98,99,99.9])
xx = int(sys.argv[2])
pxx = pxx_list[xx]
print('pxx = '+str(pxx),flush=True)

year0 = 1980 #1850
year1 = 2082
years = range(year0,(year1+1),1)
nyears = len(years)

nyears_window = 35
pxx_year = np.arange(year0,year1,1)

f_pxx = 'CESM2-LE_FWI_'+str(pxx_year[0])+'_'+str(nyears_window)+'-year_moving_high_percentiles_allens_U10_NWHemi.nc'
nc_pxx = Dataset(dir_pxx+f_pxx,'r')
lat_pxx = nc_pxx.variables['lat'][:]
lon_pxx = nc_pxx.variables['lon'][:]
lat_pxx_inds = np.where(lat_pxx>=0)[0]
lon_pxx_inds = np.where(lon_pxx>=180)[0]
lat = lat_pxx[lat_pxx_inds]
lon = lon_pxx[lon_pxx_inds]
nlat = len(lat)
nlon = len(lon)

print('reading fwi pxx', flush=True)
fwi_pxx = np.full((nyears,nlat,nlon),fill_value=-999.0)
for yy in range(0,len(pxx_year),1):
 print(pxx_year[yy], flush=True)
 f_pxx = 'CESM2-LE_FWI_'+str(pxx_year[yy])+'_'+str(nyears_window)+'-year_moving_high_percentiles_allens_U10_NWHemi.nc' 
 nc_pxx = Dataset(dir_pxx+f_pxx,'r')
 fwi_pxx[yy,:,:] = nc_pxx.variables['FWI'][xx,lat_pxx_inds,lon_pxx_inds]

files_list = [] 
for file in sorted(glob.glob(dir_fwi+'*U10.nc')):
 files_list.append(os.path.relpath(file, dir_fwi))

ens_split = [i.split('-', 1)[1] for i in files_list]
ens_bb0 = [i.split('.')[2] for i in files_list]
ens_bb = [i.split('0')[1] for i in ens_bb0]
ens_string0 = [i.split('.')[0] for i in ens_split]
ens_string1 = [i.split('.')[1] for i in ens_split]
ens_string = list(map('.'.join, zip(ens_string0, ens_string1)))
print(ens_bb[nn]+'_'+ens_string[nn], flush=True)
ens_float = np.array(ens_string).astype(float)

f_fwi0 = files_list[0]
nc_fwi0 = Dataset(dir_fwi+f_fwi0,'r')
ff_date0 = nc_fwi0.variables['date'][:]
ff_year0 = (ff_date0/10000).astype(int)

date_inds = np.where((ff_year0>=year0)&(ff_year0<=year1))[0]
ntime = len(date_inds)

date = ff_date0[date_inds]
year = ff_year0[date_inds]

f_fwi = files_list[nn]
nc_fwi = Dataset(dir_fwi+f_fwi,'r')
lat_fwi = nc_fwi.variables['lat'][:]
lon_fwi = nc_fwi.variables['lon'][:]
lat_fwi_inds = np.where(lat_fwi>=0)[0]
lon_fwi_inds = np.where(lon_fwi>=180)[0]

print('reading fwi', flush=True)
fwi = nc_fwi.variables['FWI'][date_inds,lat_fwi_inds,lon_fwi_inds]

print('broadcasting fwi pxx', flush=True)
fwi_pxx_3d = np.repeat(fwi_pxx,365,0)

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

# save labeled file information 
print('writing file', flush=True)
file_event = dir_event+'CESM2-LE_'+ens_string[nn]+'_allens_'+str(nyears_window)+'-year_moving_p'+str(pxx)+'_FWI_event_numbers_'+str(year0)+'-'+str(year1)+'_U10_NWHemi.nc'
nc_event = Dataset(file_event,'w',format='NETCDF4')
nc_event.description = 'All ensembles (1-100) p'+str(pxx)+' Fire Weather Index exceedence event numbers using moving pxx. Masked oceans. SciPy functions used: label().' 
nc_event.createDimension('date',ntime)
nc_event.createDimension('lat',nlat)
nc_event.createDimension('lon',nlon)
nc_event.createVariable('date','i4',('date',))
nc_event.createVariable('lat',lat.dtype,('lat',))
nc_event.createVariable('lon',lon.dtype,('lon',))
nc_event.createVariable('event',fwi_labeled.dtype,('date','lat','lon'))
nc_event.variables['date'][:] = date
nc_event.variables['lat'][:] = lat
nc_event.variables['lon'][:] = lon
nc_event.variables['event'][:] = fwi_labeled
nc_event.variables['lat'].units = 'degN' 
nc_event.variables['lon'].units = 'degE' 
nc_event.close()
