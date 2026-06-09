import numpy as np
from netCDF4 import Dataset 
import pandas as pd
import sys
import glob
import os
import regionmask
import xarray as xr
from datetime import datetime

# Directories
dir_fwi = '/glade/derecho/scratch/detouma/fire-precip/FWI/RH_VP_mapped/'
dir_event = '/glade/derecho/scratch/detouma/extreme_objects/fwi_objects/reg_groups/'
dir_tables = '/glade/derecho/scratch/detouma/extreme_objects/object_tables/'
dir_area = '/glade/campaign/cgd/cesm/CESM2-LE/lnd/proc/tseries/month_1/SOILWATER_10CM/'
dir_fields = '/glade/campaign/cgd/cas/detouma/fwi_objects/'
dir_vars = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/'

# set ens in bash script
ens = int(sys.argv[1])

pxx_list = np.array([90,95,98,99,99.9])
xx = int(sys.argv[2])
pxx = pxx_list[xx]
print('pxx = '+str(pxx),flush=True)

reg_groups = ['PacCoast','FourCorners']
group_states = [['CA','OR','WA'],['CO','UT','NM','AZ']]
# set rr in bash scrip
gg = int(sys.argv[3])
print('reg group = '+reg_groups[gg], flush=True)

hist_var = 'RHREFHT'
print(hist_var,flush=True)

n_ens_dist = 70

pxx_year = 1980
year0 = 1980
year1 = 2082
ndays = 365
nyears = year1-year0
years = np.arange(year0,year1,1)

event_files_list = []
for file in sorted(glob.glob(dir_event+'CESM2-LE_*_dist_VP_mapped_'+hist_var+'_to_'+str(n_ens_dist)+'ens_'+str(pxx_year)+'_p'+str(pxx)+'_FWI_event_numbers_'+str(year0)+'-'+str(year1)+'_'+reg_groups[gg]+'.nc')):
 event_files_list.append(os.path.relpath(file, dir_event))

print(event_files_list[ens], flush=True)

ens_no_string = [i.split('_')[1] for i in event_files_list]
print(ens_no_string[ens], flush=True)

# read 3d event file lat/lon
print('reading event netcdf file (using Dataset)', end = ': ')
file_event = dir_event+'CESM2-LE_'+ens_no_string[ens]+'_dist_VP_mapped_'+hist_var+'_to_'+str(n_ens_dist)+'ens_'+str(pxx_year)+'_p'+str(pxx)+'_FWI_event_numbers_'+str(year0)+'-'+str(year1)+'_'+reg_groups[gg]+'.nc'
nc_event = Dataset(file_event,'r')
lat = nc_event.variables['lat'][:]
lon = nc_event.variables['lon'][:]
nlat = len(lat)
nlon = len(lon)

# read in events
event = nc_event.variables['event'][:] 
# regions
us_states_50 = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
xr_event = xr.open_dataset(file_event)
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
mask_reg_3d = np.broadcast_to(mask_reg,event.shape)

# read in FWI
print('reading fwi netcdf file (using xarray and converting to np array)', end = ': ')

f_fwi = 'FWI_CESM2-LE_'+ens_no_string[ens]+'_historic_'+str(pxx_year)+'_dist_VP_mapped_'+hist_var+'_'+str(year0)+'0101-'+str(year1)+'1231.nc'
fwi_xr = xr.open_dataset(dir_fwi+f_fwi)
fwi_xr_sel = fwi_xr.sel(time=fwi_xr.time.dt.year.isin(np.arange(year0,year1,1)))
fwi = np.array(fwi_xr_sel.FWI)

year = np.array(fwi_xr_sel.time.dt.year)
month = np.array(fwi_xr_sel.time.dt.month)
day = np.array(fwi_xr_sel.time.dt.day)
date = year*10000+month*100+day

# season naming
season_months = np.array([[12,1,2],[3,4,5],[6,7,8],[9,10,11]])
season_names = np.array(['DJF','MAM','JJA','SON'])

season = np.zeros_like(month, dtype=season_names.dtype)

for ss in range(0,len(season_names),1):
 for mm in range(0,len(season_months[ss]),1):
  season[month==season_months[ss][mm]] = season_names[ss][:]

# cesm grid area in meters
f_area = 'b.e21.BHISTcmip6.f09_g17.LE2-1001.001.clm2.h0.SOILWATER_10CM.185001-185912.nc' # land area only, ocean is missing
nc_area = Dataset(dir_area+f_area,'r')
area_lat = nc_area.variables['lat'][:]
area_lon = nc_area.variables['lon'][:]
lat_area_inds = np.where(area_lat>=0)[0]
lon_area_inds = np.where(area_lon>=180)[0]
area = nc_area.variables['area'][lat_area_inds,lon_area_inds] # in km2 
area_units =  nc_area.variables['area'].units
area_3d = np.broadcast_to(area,fwi.shape)

# find number of events and reg events 
event_reg_nos0 = np.unique(event)
event_reg_nos = event_reg_nos0[event_reg_nos0!=0]
n_events_reg = len(event_reg_nos)

# 3d event fields
f_fields = dir_fields+'fields_3d_table_fwi_short.txt'
fields = pd.read_csv(f_fields,delimiter=',',header=0)

field_names = np.array(fields['name'])
field_types = np.array(fields['type'])
nfields = len(field_names)

field_type_dict = dict(zip(field_names, field_types))
event_3d_df = pd.DataFrame(index=range(0,n_events_reg,1),columns=field_names)
event_3d_df['event_id'] = event_reg_nos

for nn in range(0,n_events_reg,1):
 print(str(nn+1) +' out of ' + str(n_events_reg), flush = True)
 nn_ind = np.where(event==event_reg_nos[nn])
 nn_date_ind = nn_ind[0] 
 nn_start_ind = np.min(nn_date_ind)
 nn_end_ind = np.max(nn_date_ind)
 nn_days = nn_end_ind-nn_start_ind+1
 event_3d_df.loc[nn,'year'] = year[nn_start_ind]
 event_3d_df.loc[nn,'season'] = season[nn_start_ind]
 event_3d_df.loc[nn,'start_date'] = date[nn_start_ind]
 event_3d_df.loc[nn,'end_date'] = date[nn_end_ind]
 event_3d_df.loc[nn,'ndays'] = nn_days
 event_3d_df.loc[nn,'area_3d'] = np.sum(area_3d[nn_ind[0],nn_ind[1],nn_ind[2]]) # in km2
 event_3d_df.loc[nn,'avg_fwi_3d'] = np.mean(fwi[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'max_fwi_3d'] = np.max(fwi[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'area_reg_3d'] = np.ma.sum(np.ma.masked_array(area_3d,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]]) # in km2
 event_3d_df.loc[nn,'avg_fwi_reg_3d'] = np.ma.mean(np.ma.masked_array(fwi,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'max_fwi_reg_3d'] = np.ma.max(np.ma.masked_array(fwi,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 for dd in range(0,nn_days,1):
  dd_nn_ind = np.where(nn_ind[0]==(np.min(nn_ind[0])+dd))[0]
  if (dd == 0):
   event_3d_df.loc[nn,'center_lat_start'] = np.mean(lat[nn_ind[1][dd_nn_ind]]) 
   event_3d_df.loc[nn,'center_lon_start'] = np.mean(lon[nn_ind[2][dd_nn_ind]]) 
  if (dd == (nn_days-1)):
   event_3d_df.loc[nn,'center_lat_end'] = np.mean(lat[nn_ind[1][dd_nn_ind]])
   event_3d_df.loc[nn,'center_lon_end'] = np.mean(lon[nn_ind[2][dd_nn_ind]]) 

event_3d_df_nonan = event_3d_df.replace({np.nan:-999})

# set data types 
event_3d_df_nonan = event_3d_df_nonan.astype(field_type_dict)

# output event file
event_3d_df_nonan.to_csv(dir_tables+'CESM2-LE_'+ens_no_string[ens]+'_FWI_'+str(n_ens_dist)+'_ens_fixed_p'+str(pxx)+'_3d_events_'+str(year0)+'-'+str(year1)+'_'+reg_groups[gg]+'_historic_'+hist_var+'_dist_VP.txt', header=True, index=None, sep=' ', mode='w', float_format = "%.2f" )

