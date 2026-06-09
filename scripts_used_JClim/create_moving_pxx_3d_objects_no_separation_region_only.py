# Libraries
import numpy as np
from netCDF4 import Dataset 
import sys
import glob
import os
import regionmask
import xarray as xr

# Directories
dir_event = '/glade/derecho/scratch/detouma/extreme_objects/fwi_objects/'
dir_event_2d = '/glade/derecho/scratch/detouma/extreme_objects/fwi_objects/'
dir_event_3d = '/glade/derecho/scratch/detouma/extreme_objects/fwi_objects/'

# set ens in bash script
ens = int(sys.argv[1])

pxx_list = np.array([90,95,98,99,99.9])
xx = int(sys.argv[2])
pxx = pxx_list[xx]
print('pxx = '+str(pxx),flush=True)

WUS_regions = ['NorthernCA','SouthernCA','WA','OR','CO', 'UT','NM','AZ']
CA_split_lat = 37

# set rr in bash script
rr = int(sys.argv[3])
print('region = '+WUS_regions[rr], flush=True)
if (WUS_regions[rr]=='NorthernCA')|(WUS_regions[rr]=='SouthernCA'):
 reg_abbr = 'CA'
else:
 reg_abbr = WUS_regions[rr] 

year0 = 1980
year1 = 2082
ndays = 365
nyears = year1-year0+1

nyears_window = 35

files_list = []
for file in sorted(glob.glob(dir_event_3d+'CESM2-LE_*_allens_'+str(nyears_window)+'-year_moving_p'+str(pxx)+'_FWI_event_numbers_'+str(year0)+'-'+str(year1)+'_U10_NWHemi.nc')):
 files_list.append(os.path.relpath(file, dir_event_3d))

ens_no_string = [i.split('_')[1] for i in files_list]
print(ens_no_string[ens], flush=True)

# read 3d event file lat/lon
file_event_3d = dir_event_3d+'CESM2-LE_'+ens_no_string[ens]+'_allens_'+str(nyears_window)+'-year_moving_p'+str(pxx)+'_FWI_event_numbers_'+str(year0)+'-'+str(year1)+'_U10_NWHemi.nc'
nc_event_3d = Dataset(file_event_3d,'r')
lat = nc_event_3d.variables['lat'][:]
lon = nc_event_3d.variables['lon'][:]
nlat = len(lat)
nlon = len(lon)
date = nc_event_3d.variables['date'][:]
ntime = len(date)

# read in events
event_3d = nc_event_3d.variables['event'][:] 

# read 2d event file lat/lon
file_event_2d = dir_event_2d+'CESM2-LE_'+ens_no_string[ens]+'_allens_'+str(nyears_window)+'-year_moving_p'+str(pxx)+'_FWI_daily_2d_event_numbers_'+str(year0)+'-'+str(year1)+'_U10_NWHemi.nc'
nc_event_2d = Dataset(file_event_2d,'r')
event_2d = nc_event_2d.variables['event'][:]

# regions
# using code from https://stackoverflow.com/questions/70241007/mask-netcdf-using-shapefile-and-calculate-average-and-anomaly-for-all-polygons-w (user: mathause)
# load polygons of US states
us_states_50 = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
xr_event = xr.open_dataset(file_event_3d)
mask_all = us_states_50.mask_3D(xr_event)

mask_reg_lats = np.where(mask_all[mask_all['abbrevs']==reg_abbr,:,:]==True)[1]
mask_reg_lons = np.where(mask_all[mask_all['abbrevs']==reg_abbr,:,:]==True)[2]

mask_reg = np.zeros((nlat,nlon), dtype=int)
mask_reg[mask_reg_lats,mask_reg_lons] = 1

def find_nearest(array, value):
 array = np.asarray(array)
 idx = (np.abs(array - value)).argmin()
 return idx

CA_split_lat_ind = find_nearest(lat,CA_split_lat)

if WUS_regions[rr]=='NorthernCA':
 print('splitting CA - north', flush=True)
 mask_reg[CA_split_lat_ind:,:] = 0
elif  WUS_regions[rr] == 'SouthernCA':
 print('splitting CA - south', flush=True)
 mask_reg[:CA_split_lat_ind,:] = 0

mask_reg_3d = np.broadcast_to(mask_reg,event_3d.shape)

# find number of events and reg events 
event_3d_reg = np.zeros_like(event_3d,dtype=int)
event_3d_reg[:] = event_3d
event_3d_reg[mask_reg_3d==0] = 0
event_3d_reg_nos = np.unique(event_3d_reg)[1:]
n_events_3d_reg = len(event_3d_reg_nos)

event_2d_reg = np.zeros_like(event_2d,dtype=int)
event_2d_reg[:] = event_2d
event_2d_reg[mask_reg_3d==0] = 0

event_3d_reg_updated = np.zeros_like(event_3d,dtype=int)

new_event_3d_no = 1
for nn in range(0,n_events_3d_reg,1):
 no_event_flag = 0
 print(str(nn+1)+' out of ' + str(n_events_3d_reg), flush=True)
 nn_ind = np.where(event_3d==event_3d_reg_nos[nn])
 nn_date_ind = nn_ind[0]
 nn_date_ind_unique = np.unique(nn_date_ind)
 for dd in nn_date_ind_unique:
  dd_nn_ind = np.where(event_3d[dd,:,:]==event_3d_reg_nos[nn])
  event_3d_dd = event_3d[dd,dd_nn_ind[0],dd_nn_ind[1]]
  event_2d_dd_reg_nos = np.unique(event_2d_reg[dd,dd_nn_ind[0],dd_nn_ind[1]])
  event_2d_dd_reg_nos_no0 = event_2d_dd_reg_nos[event_2d_dd_reg_nos!=0]
  if len(event_2d_dd_reg_nos_no0)<1:
   no_event_flag = 1
  else:
   new_event_3d_no += no_event_flag
   no_event_flag = 0
   for nn_2d in event_2d_dd_reg_nos_no0:
    event_2d_all_ind = np.where(event_2d[dd,:,:]==nn_2d)   
    event_3d_reg_updated[dd,event_2d_all_ind[0],event_2d_all_ind[1]] = new_event_3d_no
 new_event_3d_no += 1
   
print('writing file', flush=True)
file_event = dir_event_3d+'CESM2-LE_'+ens_no_string[ens]+'_allens_moving_p'+str(pxx)+'_FWI_3d_event_numbers_'+str(year0)+'-'+str(year1)+'_'+WUS_regions[rr]+'_only_U10.nc'
nc_event = Dataset(file_event,'w',format='NETCDF4')
nc_event.description = 'All ensembles (1-100) p'+str(pxx)+' Fire Weather Index exceedence event numbers. SciPy functions used: label().'
nc_event.createDimension('date',ntime)
nc_event.createDimension('lat',nlat)
nc_event.createDimension('lon',nlon)
nc_event.createVariable('date','i4',('date',))
nc_event.createVariable('lat',lat.dtype,('lat',))
nc_event.createVariable('lon',lon.dtype,('lon',))
nc_event.createVariable('event',event_3d_reg_updated.dtype,('date','lat','lon'))
nc_event.variables['date'][:] = date
nc_event.variables['lat'][:] = lat
nc_event.variables['lon'][:] = lon
nc_event.variables['event'][:] = event_3d_reg_updated
nc_event.variables['lat'].units = 'degN'
nc_event.variables['lon'].units = 'degE'
nc_event.close()

