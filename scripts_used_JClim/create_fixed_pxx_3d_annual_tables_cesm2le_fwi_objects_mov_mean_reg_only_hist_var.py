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
dir_fwi = '/glade/campaign/cgd/cas/detouma/FWI_vars/FWI_hist_mapped/dist_mapped/'
dir_event = '/glade/campaign/cgd/cas/detouma/fwi_objects/3d_var_mapped_objects/reg_objects/dist_mapped/'
dir_tables = '/glade/derecho/scratch/detouma/extreme_objects/object_tables/'
dir_var_anoms = '/glade/campaign/cgd/cas/detouma/FWI_vars/'
dir_area = '/glade/campaign/cgd/cesm/CESM2-LE/lnd/proc/tseries/month_1/SOILWATER_10CM/'
dir_fields = '/glade/campaign/cgd/cas/detouma/fwi_objects/'

# set ens in bash script
ens = int(sys.argv[1])

pxx_list = np.array([90,95,98,99,99.9])
xx = int(sys.argv[2])
pxx = pxx_list[xx]
print('pxx = '+str(pxx),flush=True)

n_ens_dist = 100

WUS_regions = ['NorthernCA','SouthernCA','WA','OR','CO', 'UT','NM','AZ']
CA_split_lat = 37

# set rr in bash scrip
rr = int(sys.argv[3])
print('region = '+WUS_regions[rr], flush=True)
if (WUS_regions[rr]=='NorthernCA')|(WUS_regions[rr]=='SouthernCA'):
 reg_abbr = 'CA'
else:
 reg_abbr = WUS_regions[rr] 

var_ind = int(sys.argv[4])
fwi_vars = ['PRECT','TREFHTMX','RHREFHT','U10']
hist_var_select = np.zeros(len(fwi_vars), dtype='int')
hist_var_select[var_ind] = 1
hist_var = fwi_vars[var_ind]
print(hist_var,flush=True)

pxx_year = 1980
year0 = 1980
year1 = 2082
ndays = 365
nyears = year1-year0
years = np.arange(year0,year1,1)

event_files_list = []
for file in sorted(glob.glob(dir_event+'CESM2-LE_*_mapped_'+hist_var+'_to_'+str(n_ens_dist)+'_ens_'+str(pxx_year)+'_p'+str(pxx)+'_FWI_event_numbers_'+str(year0)+'-'+str(year1)+'_'+WUS_regions[rr]+'.nc')):
 event_files_list.append(os.path.relpath(file, dir_event))

print(event_files_list[ens], flush=True)

ens_no_string = [i.split('_')[1] for i in event_files_list]
print(ens_no_string[ens], flush=True)

# read 3d event file lat/lon
file_event = dir_event+'CESM2-LE_'+ens_no_string[ens]+'_mapped_'+hist_var+'_to_'+str(n_ens_dist)+'_ens_'+str(pxx_year)+'_p'+str(pxx)+'_FWI_event_numbers_'+str(year0)+'-'+str(year1)+'_'+WUS_regions[rr]+'.nc'
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

mask_reg_3d = np.broadcast_to(mask_reg,event.shape)

# read in FWI
f_fwi = 'FWI_CESM2-LE_'+ens_no_string[ens]+'_historic_'+str(pxx_year)+'_distribution_mapped_'+hist_var+'_'+str(year0)+'0101-'+str(year1)+'1231.nc'
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

# FWI VARIABLES
dir_dict = {}
fname_dict = {}
ens_split_dict = {}
ens_split_len_dict = {}
for vv in range(0,len(fwi_vars),1):
 if (hist_var_select[vv]==0):
  dir_dict[fwi_vars[vv]] = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/'+fwi_vars[vv]+'/'
  fname_dict[fwi_vars[vv]] = '*'
  ens_split_dict[fwi_vars[vv]] = '-'
  ens_split_len_dict[fwi_vars[vv]] = 1
 else:
  dir_dict[fwi_vars[vv]] = '/glade/campaign/cgd/cas/detouma/FWI_vars/'+fwi_vars[vv]+'/'
  fname_dict[fwi_vars[vv]] = 'CESM2-LE_*_'+hist_var+'_mapped_to_'+str(pxx_year)+'_'+str(n_ens_dist)+'_ens_distribution_*'
  ens_split_dict[fwi_vars[vv]] = '_'
  ens_split_len_dict[fwi_vars[vv]] = -1

# function to preprocess variable datasets
def xr_coord_sel(ds):
 return ds.sel(lon=slice(180,360), lat=slice(0,90))

#read in precipitation
print('read in pr', flush=True)
pr_files_list = []
for file in sorted(glob.glob(dir_dict['PRECT']+fname_dict['PRECT']+'.nc')):
 pr_files_list.append(os.path.relpath(file, dir_dict['PRECT']))

pr_ens_split = [i.split(ens_split_dict['PRECT'],ens_split_len_dict['PRECT'])[1][0:8] for i in pr_files_list]
pr_ens_unique = sorted(list(set(pr_ens_split)))
pr_ens_string = pr_ens_unique[ens]
pr_files_inds = np.where(np.array(pr_ens_split)==pr_ens_string)[0]
pr_files_ens = list(dir_dict['PRECT']+pr_files_list[i] for i in list(pr_files_inds))

if len(pr_files_ens)>1:
 xr_pr = xr.open_mfdataset(pr_files_ens)
else:
 xr_pr = xr.open_dataset(pr_files_ens[0])

xr_pr_sel0 = xr_pr.sel(time=xr_pr.time.dt.year.isin(range(year0,year1,1)), lat=slice(0,90), lon=slice(180,360))
xr_pr_sel = xr_pr_sel0.sel(time=~xr_pr_sel0.get_index("time").duplicated())
pr = np.array(xr_pr_sel.PRECT*86400*1000)

#read in tmax
print('read in tmax', flush=True)
tmax_files_list = []
for file in sorted(glob.glob(dir_dict['TREFHTMX']+fname_dict['TREFHTMX']+'.nc')):
 tmax_files_list.append(os.path.relpath(file, dir_dict['TREFHTMX']))

tmax_ens_split = [i.split(ens_split_dict['TREFHTMX'],ens_split_len_dict['TREFHTMX'])[1][0:8] for i in tmax_files_list]
tmax_ens_unique = sorted(list(set(tmax_ens_split)))
tmax_ens_string = tmax_ens_unique[ens]
tmax_files_inds = np.where(np.array(tmax_ens_split)==tmax_ens_string)[0]
tmax_files_ens = list(dir_dict['TREFHTMX']+tmax_files_list[i] for i in list(tmax_files_inds))

if len(tmax_files_ens)>1:
 xr_tmax = xr.open_mfdataset(tmax_files_ens)
else:
 xr_tmax = xr.open_dataset(tmax_files_ens[0])

xr_tmax_sel0 = xr_tmax.sel(time=xr_tmax.time.dt.year.isin(range(year0,year1,1)), lat=slice(0,90), lon=slice(180,360))
xr_tmax_sel = xr_tmax_sel0.sel(time=~xr_tmax_sel0.get_index("time").duplicated())

tmax = np.array(xr_tmax_sel.TREFHTMX)

#read in rh
print('read in rh', flush=True)
rh_files_list = []
for file in sorted(glob.glob(dir_dict['RHREFHT']+fname_dict['RHREFHT']+'.nc')):
 rh_files_list.append(os.path.relpath(file, dir_dict['RHREFHT']))

rh_ens_split = [i.split(ens_split_dict['RHREFHT'],ens_split_len_dict['RHREFHT'])[1][0:8] for i in rh_files_list]
rh_ens_unique = sorted(list(set(rh_ens_split)))
rh_ens_string = rh_ens_unique[ens]
rh_files_inds = np.where(np.array(rh_ens_split)==rh_ens_string)[0]
rh_files_ens = list(dir_dict['RHREFHT']+rh_files_list[i] for i in list(rh_files_inds))

if len(rh_files_ens)>1:
 xr_rh= xr.open_mfdataset(rh_files_ens)
else:
 xr_rh = xr.open_dataset(rh_files_ens[0])

xr_rh_sel0 = xr_rh.sel(time=xr_rh.time.dt.year.isin(range(year0,year1,1)), lat=slice(0,90), lon=slice(180,360))
xr_rh_sel = xr_rh_sel0.sel(time=~xr_rh_sel0.get_index("time").duplicated())
rh = np.array(xr_rh_sel.RHREFHT)

#read in wind
print('read in wind', flush=True)
wind_files_list = []
for file in sorted(glob.glob(dir_dict['U10']+fname_dict['U10']+'.nc')):
 wind_files_list.append(os.path.relpath(file, dir_dict['U10']))

wind_ens_split = [i.split(ens_split_dict['U10'],ens_split_len_dict['U10'])[1][0:8] for i in wind_files_list]
wind_ens_unique = sorted(list(set(wind_ens_split)))
wind_ens_string = wind_ens_unique[ens]
wind_files_inds = np.where(np.array(wind_ens_split)==wind_ens_string)[0]
wind_files_ens = list(dir_dict['U10']+wind_files_list[i] for i in list(wind_files_inds))

if len(wind_files_ens)>1:
 xr_wind = xr.open_mfdataset(wind_files_ens)
else:
 xr_wind = xr.open_dataset(wind_files_ens[0])

xr_wind_sel0 = xr_wind.sel(time=xr_wind.time.dt.year.isin(range(year0,year1,1)), lat=slice(0,90), lon=slice(180,360))
xr_wind_sel = xr_wind_sel0.sel(time=~xr_wind_sel0.get_index("time").duplicated())
wind = np.array(xr_wind_sel.U10)

#calculate precipitation anomaly
print('calculate pr anomaly', flush=True)
f_pr_mean = 'PRECT/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_PRECT_'+str(year0)+'.nc'
xr_pr_mean = xr.open_dataset(dir_var_anoms + f_pr_mean)
xr_pr_mean_sel = xr_pr_mean.sel(lat=slice(0,90), lon=slice(180,360))
pr_mean_365 = np.array(xr_pr_mean_sel.PRECT*86400 * 1000)
pr_mean = np.tile(pr_mean_365,[nyears,1,1])
pr_anom = pr-pr_mean

#calculate tmax anomaly
print('calculate tmax anomaly', flush=True)
f_tmax_mean = 'TREFHTMX/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_TREFHTMX_'+str(year0)+'_NWHemi.nc'
xr_tmax_mean = xr.open_dataset(dir_var_anoms + f_tmax_mean)
xr_tmax_mean_sel = xr_tmax_mean.sel(lat=slice(0,90), lon=slice(180,360))
tmax_mean_365 = np.array(xr_tmax_mean_sel.TREFHTMX)
tmax_mean = np.tile(tmax_mean_365,[nyears,1,1])
tmax_anom = tmax-tmax_mean

#calculate wind anomaly
print('calculate wind anomaly', flush=True)
f_wind_mean = 'U10/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_U10_'+str(year0)+'_NWHemi.nc'
xr_wind_mean = xr.open_dataset(dir_var_anoms + f_wind_mean)
xr_wind_mean_sel = xr_wind_mean.sel(lat=slice(0,90), lon=slice(180,360))
wind_mean_365 = np.array(xr_wind_mean_sel.U10)
wind_mean = np.tile(wind_mean_365,[nyears,1,1])
wind_anom = wind-wind_mean

#calculate rh anomaly
print('calculate rh anomaly', flush=True)
f_rh_mean = 'RHREFHT/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_RHREFHT_'+str(year0)+'_NWHemi.nc'
xr_rh_mean = xr.open_dataset(dir_var_anoms + f_rh_mean)
xr_rh_mean_sel = xr_rh_mean.sel(lat=slice(0,90), lon=slice(180,360))
rh_mean_365 = np.array(xr_rh_mean_sel.RHREFHT)
rh_mean = np.tile(rh_mean_365,[nyears,1,1])
rh_anom = rh-rh_mean

#preprocessing function for reading multiyear files
def preprocess(ds):
    return ds.expand_dims(year = [datetime.now()])

#calculate precipitation anomaly
print('calculate pr mov anomaly', flush=True)
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
event_3d_df_nonan.to_csv(dir_tables+'CESM2-LE_'+ens_no_string[ens]+'_FWI_'+str(n_ens_dist)+'_ens_fixed_p'+str(pxx)+'_3d_events_'+str(year0)+'-'+str(year1)+'_'+WUS_regions[rr]+'_historic_'+hist_var+'_dist.txt', header=True, index=None, sep=' ', mode='w', float_format = "%.2f" )

