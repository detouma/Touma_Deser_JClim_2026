# Libraries
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
dir_fwi = '/glade/campaign/cgd/cas/detouma/FWI_CESM2LE/'
dir_event = '/glade/derecho/scratch/detouma/extreme_objects/fwi_objects/'
dir_tables = '/glade/derecho/scratch/detouma/extreme_objects/object_tables/'
dir_var_anoms = '/glade/campaign/cgd/cas/detouma/FWI_vars/'
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

year0 = 1980
year1 = 2082
ndays = 365
nyears = year1-year0+1
years = np.arange(year0,(year1+1),1)

event_files_list = []
for file in sorted(glob.glob(dir_event+'CESM2-LE_*_allens_fixed_p'+str(pxx)+'_FWI_3d_event_numbers_'+str(year0)+'-'+str(year1)+'_'+reg_groups[gg]+'_U10.nc')):
 event_files_list.append(os.path.relpath(file, dir_event))

ens_no_string = [i.split('_')[1] for i in event_files_list]
print(ens_no_string[ens], flush=True)

# check if file already exists and exit if it does!
if os.path.isfile(dir_tables+'CESM2-LE_'+ens_no_string[ens]+'_FWI_allens_fixed_p'+str(pxx)+'_3d_events_'+str(year0)+'-'+str(year1)+'_'+reg_groups[gg]+'_U10.txt'):
  print('file already exists for '+ens_no_string[ens]+' ' + reg_groups[gg])
  exit()

# read 3d event file lat/lon
file_event = dir_event+'CESM2-LE_'+ens_no_string[ens]+'_allens_fixed_p'+str(pxx)+'_FWI_3d_event_numbers_'+str(year0)+'-'+str(year1)+'_'+reg_groups[gg]+'_U10.nc'
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
f_fwi = glob.glob(dir_fwi+'*'+ens_no_string[ens]+'*.nc')[0]
nc_fwi = Dataset(f_fwi,'r')

ff_date0 = nc_fwi.variables['date'][:]
ff_year0 = (ff_date0/10000).astype(int)
date_inds = np.where((ff_year0>=year0)&(ff_year0<=year1))[0]
ntime = len(date_inds)

date = ff_date0[date_inds]
year = ff_year0[date_inds]

lat_fwi = nc_fwi.variables['lat'][:]
lon_fwi = nc_fwi.variables['lon'][:]
lat_fwi_inds = np.where(lat_fwi>=0)[0]
lon_fwi_inds = np.where(lon_fwi>=180)[0]
fwi = nc_fwi.variables['FWI'][date_inds,lat_fwi_inds,lon_fwi_inds]

# season naming
month = ((date%10000)/100).astype(int)
season_months = np.array([[12,1,2],[3,4,5],[6,7,8],[9,10,11]])
season_names = np.array(['DJF','MAM','JJA','SON'])

season = np.zeros_like(date, dtype=season_names.dtype)

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

# FWI VARIABLES
# function to preprocess variable datasets
def xr_coord_sel(ds):
 return ds.sel(lon=slice(180,360), lat=slice(0,90))

#read in precipitation
print('read in pr', flush=True)
pr_files = []
for ff in sorted(glob.glob(dir_vars+'PRECT/b.e21.B*')):
 pr_files.append(os.path.relpath(ff, dir_vars+'PRECT/'))

pr_files_ens_str0 = [i.split('-', 1)[1].split('.')[0] for i in pr_files]
pr_files_ens_str1 = [i.split('-', 1)[1].split('.')[1] for i in pr_files]
pr_files_ens_str = list(map('.'.join, zip(pr_files_ens_str0, pr_files_ens_str1)))
pr_files_ens_inds = np.where(np.array(pr_files_ens_str)==ens_no_string[ens])[0]
pr_files_ens = []
for ii in pr_files_ens_inds:
 pr_files_ens.append(pr_files[ii])

xr_pr = xr.open_mfdataset([dir_vars+'PRECT/'+x for x in pr_files_ens], preprocess=xr_coord_sel)
xr_pr_time = xr_pr.sel(time=xr_pr.time.dt.year.isin(range(year0,year1+1,1)))
pr = np.array(xr_pr_time.PRECT.sel(time=~xr_pr_time.get_index("time").duplicated())*86400*1000)

#read in tmax
print('read in tmax', flush=True)
tmax_files = []
for ff in sorted(glob.glob(dir_vars+'TREFHTMX/b.e21.B*')):
 tmax_files.append(os.path.relpath(ff, dir_vars+'TREFHTMX/'))

tmax_files_ens_str0 = [i.split('-', 1)[1].split('.')[0] for i in tmax_files]
tmax_files_ens_str1 = [i.split('-', 1)[1].split('.')[1] for i in tmax_files]
tmax_files_ens_str = list(map('.'.join, zip(tmax_files_ens_str0, tmax_files_ens_str1)))
tmax_files_ens_inds = np.where(np.array(tmax_files_ens_str)==ens_no_string[ens])[0]
tmax_files_ens = []
for ii in tmax_files_ens_inds:
 tmax_files_ens.append(tmax_files[ii])

xr_tmax = xr.open_mfdataset([dir_vars+'TREFHTMX/'+x for x in tmax_files_ens], preprocess=xr_coord_sel)
xr_tmax_time = xr_tmax.sel(time=xr_tmax.time.dt.year.isin(range(year0,year1+1,1)))
tmax = np.array(xr_tmax_time.TREFHTMX.sel(time=~xr_tmax_time.get_index("time").duplicated()))

#read in rh
print('read in rh', flush=True)
rh_files = []
for ff in sorted(glob.glob(dir_vars+'RHREFHT/b.e21.B*')):
 rh_files.append(os.path.relpath(ff, dir_vars+'RHREFHT/'))

rh_files_ens_str0 = [i.split('-', 1)[1].split('.')[0] for i in rh_files]
rh_files_ens_str1 = [i.split('-', 1)[1].split('.')[1] for i in rh_files]
rh_files_ens_str = list(map('.'.join, zip(rh_files_ens_str0, rh_files_ens_str1)))
rh_files_ens_inds = np.where(np.array(rh_files_ens_str)==ens_no_string[ens])[0]
rh_files_ens = []
for ii in rh_files_ens_inds:
 rh_files_ens.append(rh_files[ii])

xr_rh = xr.open_mfdataset([dir_vars+'RHREFHT/'+x for x in rh_files_ens], preprocess=xr_coord_sel)
xr_rh_time = xr_rh.sel(time=xr_rh.time.dt.year.isin(range(year0,year1+1,1)))
rh = np.array(xr_rh_time.RHREFHT.sel(time=~xr_rh_time.get_index("time").duplicated()))

#read in wind
print('read in wind', flush=True)
wind_files = []
for ff in sorted(glob.glob(dir_vars+'U10/b.e21.B*')):
 wind_files.append(os.path.relpath(ff, dir_vars+'U10/'))

wind_files_ens_str0 = [i.split('-', 1)[1].split('.')[0] for i in wind_files]
wind_files_ens_str1 = [i.split('-', 1)[1].split('.')[1] for i in wind_files]
wind_files_ens_str = list(map('.'.join, zip(wind_files_ens_str0, wind_files_ens_str1)))
wind_files_ens_inds = np.where(np.array(wind_files_ens_str)==ens_no_string[ens])[0]
wind_files_ens = []
for ii in wind_files_ens_inds:
 wind_files_ens.append(wind_files[ii])

xr_wind = xr.open_mfdataset([dir_vars+'U10/'+x for x in wind_files_ens], preprocess=xr_coord_sel)
xr_wind_time = xr_wind.sel(time=xr_wind.time.dt.year.isin(range(year0,year1+1,1)))
wind = np.array(xr_wind_time.U10.sel(time=~xr_wind_time.get_index("time").duplicated()))

#calculate precipitation anomaly
print('calculate pr anomaly', flush=True)
f_pr_mean = 'PRECT/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_PRECT_'+str(year0)+'.nc'
nc_pr_mean = Dataset(dir_var_anoms + f_pr_mean, 'r')
pr_mean_365 = nc_pr_mean.variables['PRECT'][:,lat_fwi_inds,lon_fwi_inds] * 86400 * 1000 # m/s -> mm/day
pr_mean = np.tile(pr_mean_365,[nyears,1,1])
pr_anom = pr-pr_mean

#calculate tmax anomaly
print('calculate tmax anomaly', flush=True)
f_tmax_mean = 'TREFHTMX/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_TREFHTMX_'+str(year0)+'_NWHemi.nc'
nc_tmax_mean = Dataset(dir_var_anoms + f_tmax_mean, 'r')
tmax_mean_365 = nc_tmax_mean.variables['TREFHTMX'][:] 
tmax_mean = np.tile(tmax_mean_365,[nyears,1,1])
tmax_anom = tmax-tmax_mean

#calculate wind anomaly
print('calculate wind anomaly', flush=True)
f_wind_mean = 'U10/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_U10_'+str(year0)+'_NWHemi.nc'
nc_wind_mean = Dataset(dir_var_anoms + f_wind_mean, 'r')
wind_mean_365 = nc_wind_mean.variables['U10'][:] 
wind_mean = np.tile(wind_mean_365,[nyears,1,1])
wind_anom = wind-wind_mean

#calculate rh anomaly
print('calculate rh anomaly', flush=True)
f_rh_mean = 'RHREFHT/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_RHREFHT_'+str(year0)+'_NWHemi.nc'
nc_rh_mean = Dataset(dir_var_anoms + f_rh_mean, 'r')
rh_mean_365 = nc_rh_mean.variables['RHREFHT'][:] 
rh_mean = np.tile(rh_mean_365,[nyears,1,1])
rh_anom = rh-rh_mean

#preprocessing function for reading multiyear files
def preprocess(ds):
    return ds.expand_dims(year = [datetime.now()])

#calculate precipitation anomaly
print('calculate pr mov  anomaly', flush=True)
pr_mov_mean_files_list = []

for yy in years:
    pr_mov_mean_files_list.append(dir_var_anoms+'PRECT/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_PRECT_'+str(yy)+'.nc')

pr_mov_mean_xr = xr.open_mfdataset(pr_mov_mean_files_list, preprocess=preprocess)
pr_mov_mean_xr['year'] =  years
pr_mov_mean_doy = pr_mov_mean_xr.PRECT.sel(lon=slice(180,360), lat=slice(0,90))* 86400 * 1000
pr_mov_mean = np.reshape(np.array(pr_mov_mean_doy),newshape=((nyears*365),len(pr_mov_mean_doy.lat),len(pr_mov_mean_doy.lon)))
pr_mov_anom = pr - pr_mov_mean

#calculate tmax mov anomaly
print('calculate tmax mov anomaly', flush=True)
tmax_mov_mean_files_list = []

for yy in years:
    tmax_mov_mean_files_list.append(dir_var_anoms+'TREFHTMX/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_TREFHTMX_'+str(yy)+'_NWHemi.nc')

tmax_mov_mean_xr = xr.open_mfdataset(tmax_mov_mean_files_list, preprocess=preprocess)
tmax_mov_mean_xr['year'] =  years
tmax_mov_mean_doy = tmax_mov_mean_xr.TREFHTMX
tmax_mov_mean = np.reshape(np.array(tmax_mov_mean_doy),newshape=((nyears*365),len(tmax_mov_mean_doy.lat),len(tmax_mov_mean_doy.lon)))
tmax_mov_anom = tmax - tmax_mov_mean

#calculate wind mov anomaly
print('calculate wind mov anomaly', flush=True)
wind_mov_mean_files_list = []

for yy in years:
    wind_mov_mean_files_list.append(dir_var_anoms+'U10/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_U10_'+str(yy)+'_NWHemi.nc')

wind_mov_mean_xr = xr.open_mfdataset(wind_mov_mean_files_list, preprocess=preprocess)
wind_mov_mean_xr['year'] =  years
wind_mov_mean_doy = wind_mov_mean_xr.U10.sel(lon=slice(180,360), lat=slice(0,90))
wind_mov_mean = np.reshape(np.array(wind_mov_mean_doy),newshape=((nyears*365),len(wind_mov_mean_doy.lat),len(wind_mov_mean_doy.lon)))
wind_mov_anom = wind - wind_mov_mean

#calculate rh mov anomaly
print('calculate rh mov anomaly', flush=True)
rh_mov_mean_files_list = []

for yy in years:
    rh_mov_mean_files_list.append(dir_var_anoms+'RHREFHT/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_RHREFHT_'+str(yy)+'_NWHemi.nc')

rh_mov_mean_xr = xr.open_mfdataset(rh_mov_mean_files_list, preprocess=preprocess)
rh_mov_mean_xr['year'] =  years
rh_mov_mean_doy = rh_mov_mean_xr.RHREFHT
rh_mov_mean = np.reshape(np.array(rh_mov_mean_doy),newshape=((nyears*365),len(rh_mov_mean_doy.lat),len(rh_mov_mean_doy.lon)))
rh_mov_anom = rh - rh_mov_mean

# find number of events and reg events 
event_reg_nos0 = np.unique(event)
event_reg_nos = event_reg_nos0[event_reg_nos0!=0]
n_events_reg = len(event_reg_nos)

# 3d event fields
f_fields = dir_fields+'fields_3d_table_fwi.txt'
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
 event_3d_df.loc[nn,'pr_anom_3d'] = np.mean(pr_anom[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'tmax_anom_3d'] = np.mean(tmax_anom[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'wind_anom_3d'] = np.mean(wind_anom[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'rh_anom_3d'] = np.mean(rh_anom[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'pr_mov_anom_3d'] = np.mean(pr_mov_anom[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'tmax_mov_anom_3d'] = np.mean(tmax_mov_anom[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'wind_mov_anom_3d'] = np.mean(wind_mov_anom[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'rh_mov_anom_3d'] = np.mean(rh_mov_anom[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'pr_3d'] = np.mean(pr[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'tmax_3d'] = np.mean(tmax[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'wind_3d'] = np.mean(wind[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'rh_3d'] = np.mean(rh[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'area_reg_3d'] = np.ma.sum(np.ma.masked_array(area_3d,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]]) # in km2
 event_3d_df.loc[nn,'avg_fwi_reg_3d'] = np.ma.mean(np.ma.masked_array(fwi,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'max_fwi_reg_3d'] = np.ma.max(np.ma.masked_array(fwi,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'pr_anom_reg_3d'] = np.ma.mean(np.ma.masked_array(pr_anom,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'tmax_anom_reg_3d'] = np.ma.mean(np.ma.masked_array(tmax_anom,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'wind_anom_reg_3d'] = np.ma.mean(np.ma.masked_array(wind_anom,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'rh_anom_reg_3d'] = np.ma.mean(np.ma.masked_array(rh_anom,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'pr_mov_anom_reg_3d'] = np.ma.mean(np.ma.masked_array(pr_mov_anom,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'tmax_mov_anom_reg_3d'] = np.ma.mean(np.ma.masked_array(tmax_mov_anom,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'wind_mov_anom_reg_3d'] = np.ma.mean(np.ma.masked_array(wind_mov_anom,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'rh_mov_anom_reg_3d'] = np.ma.mean(np.ma.masked_array(rh_mov_anom,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'pr_reg_3d'] = np.ma.mean(np.ma.masked_array(pr,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'tmax_reg_3d'] = np.ma.mean(np.ma.masked_array(tmax,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'wind_reg_3d'] = np.ma.mean(np.ma.masked_array(wind,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
 event_3d_df.loc[nn,'rh_reg_3d'] = np.ma.mean(np.ma.masked_array(rh,mask=(mask_reg_3d==0))[nn_ind[0],nn_ind[1],nn_ind[2]])
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
event_3d_df_nonan.to_csv(dir_tables+'CESM2-LE_'+ens_no_string[ens]+'_FWI_allens_fixed_p'+str(pxx)+'_3d_events_'+str(year0)+'-'+str(year1)+'_'+reg_groups[gg]+'_U10.txt', header=True, index=None, sep=' ', mode='w', float_format = "%.2f" )

