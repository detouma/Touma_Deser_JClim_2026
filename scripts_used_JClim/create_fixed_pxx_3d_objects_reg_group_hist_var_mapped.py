# Libraries	
import numpy as np
import sys
import glob
import os
import regionmask
import xarray as xr

# Directories
dir_event_2d = '/glade/derecho/scratch/detouma/extreme_objects/fwi_objects/NWHemi_2d/'
dir_event_3d = '/glade/derecho/scratch/detouma/extreme_objects/fwi_objects/NWHemi_3d/'
dir_reg_group = '/glade/derecho/scratch/detouma/extreme_objects/fwi_objects/reg_groups/dist_mapped/'
# dir_event_3d = '/glade/campaign/cgd/cas/detouma/fwi_objects/3d_var_mapped_objects/dist_mapped/'
# dir_event_2d = '/glade/campaign/cgd/cas/detouma/fwi_objects/2d_objects/'

# set ens in bash script
ens = int(sys.argv[1])

pxx_list = np.array([90,95,98,99,99.9])
xx = int(sys.argv[2])
pxx = pxx_list[xx]
print('pxx = '+str(pxx),flush=True)

reg_groups = ['PacCoast','FourCorners']
group_states = [['CA','OR','WA'],['CO','UT','NM','AZ']]
# set gg in bash scrip
gg = int(sys.argv[3])
print('region group = '+reg_groups[gg], flush=True)

vv = int(sys.argv[4])
var_name = ['PRECT','TREFHTMX','RHREFHT','U10'][vv]
print(var_name, flush=True)

year0 = 1980
year1 = 2082
ndays = 365
nyears = year1-year0+1

pxx_year = 1980

n_ens_dist = 100

files_list = []
for file in sorted(glob.glob(dir_event_3d+'CESM2-LE_*_mapped_'+var_name+'_to_'+str(n_ens_dist)+'_ens_'+str(pxx_year)+'_p'+str(pxx)+'_FWI_event_numbers_'+str(year0)+'-'+str(year1)+'_NWHemi.nc')):
 files_list.append(os.path.relpath(file, dir_event_3d))

ens_no_string = [i.split('_')[1] for i in files_list]
print(ens_no_string[ens], flush=True)

# read 3d event file lat/lon
file_event_3d = dir_event_3d+'CESM2-LE_'+ens_no_string[ens]+'_mapped_'+var_name+'_to_'+str(n_ens_dist)+'_ens_'+str(pxx_year)+'_p'+str(pxx)+'_FWI_event_numbers_'+str(year0)+'-'+str(year1)+'_NWHemi.nc'
event_3d_xr = xr.open_dataset(file_event_3d)
event_3d = np.array(event_3d_xr.event)

lat = event_3d_xr.lat
lon = event_3d_xr.lon
time = event_3d_xr.time

nlat = len(lat)
nlon = len(lon)
ntime = len(time)

# read 2d event file lat/lon
file_event_2d = dir_event_2d+'CESM2-LE_'+ens_no_string[ens]+'_mapped_'+var_name+'_to_'+str(n_ens_dist)+'_ens_'+str(pxx_year)+'_p'+str(pxx)+'_FWI_daily_2d_event_numbers_'+str(year0)+'-'+str(year1)+'_NWHemi.nc'
event_2d_xr = xr.open_dataset(file_event_2d)
event_2d = np.array(event_2d_xr.event)

# regions
# using code from https://stackoverflow.com/questions/70241007/mask-netcdf-using-shapefile-and-calculate-average-and-anomaly-for-all-polygons-w (user: mathause)
# load polygons of US states
us_states_50 = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
xr_event = xr.open_dataset(file_event_3d)
mask_all = us_states_50.mask_3D(xr_event)

mask_group_lats = []
mask_group_lons = []

for ss in group_states[gg]:
 mask_ss_lats = np.where(mask_all[mask_all['abbrevs']==ss,:,:]==True)[1]
 mask_ss_lons = np.where(mask_all[mask_all['abbrevs']==ss,:,:]==True)[2]
 mask_group_lats = mask_group_lats+list(mask_ss_lats)
 mask_group_lons = mask_group_lons+list(mask_ss_lons)

mask_reg = np.zeros((nlat,nlon), dtype=int)
mask_reg[mask_group_lats,mask_group_lons] = 1
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
   #print('(no event flag on)')
  else:
   new_event_3d_no += no_event_flag
   no_event_flag = 0
   #print('(event no: '+str(new_event_3d_no)+ ', no event flag off)')
   for nn_2d in event_2d_dd_reg_nos_no0:
    event_2d_all_ind = np.where(event_2d[dd,:,:]==nn_2d)   
    event_3d_reg_updated[dd,event_2d_all_ind[0],event_2d_all_ind[1]] = new_event_3d_no
 new_event_3d_no += 1
 #print('\n')
   
print('converting to xarray', flush=True)
xr_event = xr.DataArray(event_3d_reg_updated, name='event', dims=['time','lat','lon'], coords = dict(time=time, lat=lat, lon=lon))
xr_event.attrs['description'] = 'Extreme fire weather events for '+reg_groups[gg]+' using historically mapped '+var_name+' exceeding the historic p'+str(pxx)+' threshold'
print('writing xarray to netcdf' , flush=True)
xr_event.to_netcdf(dir_reg_group+'CESM2-LE_'+ens_no_string[ens]+'_mapped_'+var_name+'_to_'+str(n_ens_dist)+'_ens_'+str(pxx_year)+'_p'+str(pxx)+'_FWI_event_numbers_'+str(year0)+'-'+str(year1)+'_'+reg_groups[gg]+'.nc')

