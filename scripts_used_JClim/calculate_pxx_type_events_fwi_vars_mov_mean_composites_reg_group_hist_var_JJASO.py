import numpy as np
import sys
import pandas as pd
import xarray as xr
import glob
import os
from datetime import datetime

dir_fwi = '/glade/campaign/cgd/cas/detouma/FWI_vars/FWI_hist_mapped/dist_mapped/'
dir_obj = '/glade/campaign/cgd/cas/detouma/fwi_objects/3d_var_mapped_objects/grp_objects/dist_mapped/'
dir_tables = '/glade/campaign/cgd/cas/detouma/fwi_objects/object_tables/grp_tables/dist_mapped/'
dir_vars = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/'
dir_var_anoms = '/glade/campaign/cgd/cas/detouma/FWI_vars/'
dir_comps = '/glade/derecho/scratch/detouma/extreme_objects/composites/'

nn = int(sys.argv[1])
gg = int(sys.argv[2])
var_ind = int(sys.argv[3])

print('nn = '+str(nn)+', gg = '+str(gg)+', var_ind = '+str(var_ind), flush=True)

fwi_vars = ['PRECT','TREFHTMX','RHREFHT','U10']
hist_var_select = np.zeros(len(fwi_vars), dtype='int')
hist_var_select[var_ind] = 1
hist_var = fwi_vars[var_ind]
print(hist_var,flush=True)

pxx_type = 'fixed'

pxx_year0 = 1980
pxx_year1 = 2014
nyears_window = 35
pxx = 99.0
pxx_string =  str(pxx)

reg_groups = ['PacCoast','FourCorners']

year0 = 1980
year1 = 2082 #2100
years = np.arange(year0,year1,1)
nyears = len(years)

season_months = [6, 7, 8, 9, 10]
season = 'JJASO'

pyear0 = [1980,2015,2050]
pyear1 = [2014,2050,2081]
nyears_period = np.array(pyear1)-np.array(pyear0)+1
nperiods = len(pyear0)
pxx_year0 = 1980
pxx_year1 = 2014
period_str = ['-'.join(i) for i in zip([str(i) for i in pyear0],[str(i) for i in pyear1])]

n_ens_dist = 100

files_3d_list = []

print('CESM2-LE_*_FWI_'+str(n_ens_dist)+'_ens_'+pxx_type+'_p'+pxx_string+'_3d_events_'+str(year0)+'-'+str(year1)+'_*_historic_'+hist_var+'_dist.txt')
for file in sorted(glob.glob(dir_tables+'CESM2-LE_*_FWI_'+str(n_ens_dist)+'_ens_'+pxx_type+'_p'+pxx_string+'_3d_events_'+str(year0)+'-'+str(year1)+'_*_historic_'+hist_var+'_dist.txt')):
    files_3d_list.append(os.path.relpath(file, dir_tables))

nfiles_3d = len(files_3d_list)

ens_3d = np.array([i.split('_')[1] for i in files_3d_list])
reg_3d =  [i.split('_')[10] for i in files_3d_list]
files_df = pd.DataFrame({'ens':ens_3d,'reg':reg_3d,'filename':files_3d_list})

events_3d_list = []

ens_unique = files_df['ens'].unique()

ens_string = ens_unique[nn]

files_ens_df = files_df[files_df['ens'] == ens_string].reset_index()

events_3d_list = []

for ff in range(0,len(files_ens_df),1):
    events_3d_ff = pd.read_csv(dir_tables+files_ens_df['filename'][ff],header=0,sep=' ')
    events_3d_ff['region'] = files_ens_df['reg'][ff]
    events_3d_list.append(events_3d_ff)
        
all_events_3d = pd.concat(events_3d_list, ignore_index=True)

all_events_3d = all_events_3d.replace(-999.0, np.nan)

all_events_3d['period'] = ''
for pp in range(0,nperiods,1):
    all_events_3d.loc[
        (all_events_3d['year']>=pyear0[pp])&(all_events_3d['year']<=pyear1[pp]),'period'] = str(pyear0[pp])+'-'+str(pyear1[pp])

all_events_3d['month'] = (all_events_3d.start_date/100).astype(int)-(all_events_3d.year*100).astype(int)

chars = ['nevents', 'ndays', 'area' ,'max_fwi', 'avg_fwi', 'pr_anom', 'tmax_anom', 'rh_anom', 'wind_anom', 'pr_mov_anom', 'tmax_mov_anom', 'rh_mov_anom', 'wind_mov_anom', 'pr', 'tmax', 'rh', 'wind']

# read in FWI
f_fwi = 'FWI_CESM2-LE_'+ens_string+'_historic_'+str(pxx_year0)+'_distribution_mapped_'+hist_var+'_'+str(year0)+'0101-'+str(year1)+'1231.nc'
fwi_xr = xr.open_dataset(dir_fwi+f_fwi)
fwi_xr_sel = fwi_xr.sel(time=fwi_xr.time.dt.year.isin(np.arange(year0,year1,1)))
fwi = np.array(fwi_xr_sel.FWI)

year = np.array(fwi_xr_sel.time.dt.year)
month = np.array(fwi_xr_sel.time.dt.month)
day = np.array(fwi_xr_sel.time.dt.day)
date = year*10000+month*100+day

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
  fname_dict[fwi_vars[vv]] = 'CESM2-LE_*_'+hist_var+'_mapped_to_'+str(pxx_year0)+'_'+str(n_ens_dist)+'_ens_distribution_*'
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
pr_ens_string = pr_ens_unique[nn]
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
tmax_ens_string = tmax_ens_unique[nn]
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
rh_ens_string = rh_ens_unique[nn]
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
wind_ens_string = wind_ens_unique[nn]
wind_files_inds = np.where(np.array(wind_ens_split)==wind_ens_string)[0]
wind_files_ens = list(dir_dict['U10']+wind_files_list[i] for i in list(wind_files_inds))

if len(wind_files_ens)>1:
 xr_wind = xr.open_mfdataset(wind_files_ens)
else:
 xr_wind = xr.open_dataset(wind_files_ens[0])

xr_wind_sel0 = xr_wind.sel(time=xr_wind.time.dt.year.isin(range(year0,year1,1)), lat=slice(0,90), lon=slice(180,360))
xr_wind_sel = xr_wind_sel0.sel(time=~xr_wind_sel0.get_index("time").duplicated())
wind = np.array(xr_wind_sel.U10)

#preprocessing function for reading multiyear files
def preprocess(ds):
    return ds.expand_dims(year = [datetime.now()])

# MOVING ANOMALIES
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

#calculate precipitation mov anomaly
print('calculate wind mov  anomaly', flush=True)
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

#calculate precipitation anomaly
print('calculate pr anomaly', flush=True)
f_pr_mean = dir_var_anoms+'PRECT/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_PRECT_'+str(year0)+'.nc'
xr_pr_mean = xr.open_dataset(f_pr_mean)
pr_mean_365 = np.array(xr_pr_mean.PRECT.sel(lon=slice(180,360), lat=slice(0,90)))* 86400 * 1000 # m/s -> mm/day
pr_mean = np.tile(pr_mean_365,[nyears,1,1])
pr_anom = pr-pr_mean

#calculate tmax anomaly
print('calculate tmax anomaly', flush=True)
f_tmax_mean = dir_var_anoms+'TREFHTMX/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_TREFHTMX_'+str(year0)+'_NWHemi.nc'
xr_tmax_mean = xr.open_dataset(f_tmax_mean)
tmax_mean_365 = np.array(xr_tmax_mean.TREFHTMX) 
tmax_mean = np.tile(tmax_mean_365,[nyears,1,1])
tmax_anom = tmax-tmax_mean

#calculate wind anomaly
print('calculate wind anomaly', flush=True)
f_wind_mean =  dir_var_anoms+'U10/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_U10_'+str(year0)+'_NWHemi.nc'
xr_wind_mean = xr.open_dataset(f_wind_mean)
wind_mean_365 = np.array(xr_wind_mean.U10.sel(lon=slice(180,360), lat=slice(0,90)))
wind_mean = np.tile(wind_mean_365,[nyears,1,1])
wind_anom = wind-wind_mean

#calculate rh anomaly
print('calculate rh anomaly', flush=True)
f_rh_mean = dir_var_anoms+'RHREFHT/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_RHREFHT_'+str(year0)+'_NWHemi.nc'
xr_rh_mean = xr.open_dataset(f_rh_mean)
rh_mean_365 = np.array(xr_rh_mean.RHREFHT) 
rh_mean = np.tile(rh_mean_365,[nyears,1,1])
rh_anom = rh-rh_mean

# object files
print('reading object file', flush=True)
object_file = 'CESM2-LE_'+ens_string+'_mapped_'+hist_var+'_to_'+str(n_ens_dist)+'_ens_'+str(pxx_year0)+'_p'+pxx_string+'_FWI_event_numbers_'+str(year0)+'-'+str(year1)+'_'+reg_groups[gg]+'.nc'
xr_obj = xr.open_dataset(dir_obj+object_file)
nlat = len(xr_obj.lat)
nlon = len(xr_obj.lon)

obj_all = xr_obj.event

for pp in range(0,nperiods,1):
	print(reg_groups[gg] + ' | ' + ' | ' + period_str[pp] + ' | ' + hist_var, flush=True)
	event_subset_df = all_events_3d[(all_events_3d['region']==reg_groups[gg])&(all_events_3d['month'].isin(season_months))&(all_events_3d['period']==period_str[pp])].reset_index()
	n_events_subset = len(event_subset_df)
	events_char_pp = np.full(fill_value = -999.0, shape=(len(chars),nlat,nlon))
	pp_event_mask = np.zeros(obj_all.shape,dtype=int) #time x lat x lon - where events occur in time and space
	pp_nevents = np.zeros((nlat,nlon),dtype=int) #lat x lon - add up numper of events for each grid point
	pp_events_ndays = np.zeros((n_events_subset,nlat,nlon),dtype=int) #events x lat x lon - number of days for each event at each grid point
	pp_events_area = np.zeros((n_events_subset,nlat,nlon),dtype=int) # events x lat x lon - area of each event at each grod point
	for nn in range(0,n_events_subset,1):
		if nn%5==0:
			print(str(nn)+' out of '+str(n_events_subset), end= ', ', flush=True)
		event_id_nn = event_subset_df['event_id'][nn]
		nn_inds = np.where(obj_all==event_id_nn)
		pp_event_mask[nn_inds] = 1
		pp_nevents[nn_inds[1],nn_inds[2]] += 1
		pp_events_ndays[nn,nn_inds[1],nn_inds[2]] = event_subset_df['ndays'][nn]
		pp_events_area[nn,nn_inds[1],nn_inds[2]] = event_subset_df['area_3d'][nn] 
	print('nevents', flush=True) 
	events_char_pp[0,:,:] = pp_nevents
	print('ndays', flush=True) 
	pp_events_ndays = np.ma.masked_array(pp_events_ndays, mask=(pp_events_ndays==0))
	events_char_pp[1,:,:] = np.ma.sum(pp_events_ndays,axis=0) # ndays
	print('area', flush=True) 
	pp_events_area = np.ma.masked_array(pp_events_area, mask=(pp_events_area==0))
	events_char_pp[2,:,:] = np.ma.mean(pp_events_area,axis=0) #area
	print('max_fwi', flush=True) 
	events_char_pp[3,:,:] = np.ma.max(np.ma.masked_array(fwi,mask=(pp_event_mask==0)),axis=0)
	print('avg_fwi', flush=True) 
	events_char_pp[4,:,:] = np.ma.mean(np.ma.masked_array(fwi,mask=(pp_event_mask==0)),axis=0)
	for cc in range(5,len(chars),1):
		print(chars[cc], flush=True) 
		events_char_pp[cc,:,:] = np.ma.mean(np.ma.masked_array(eval(chars[cc]),mask=(pp_event_mask==0)),axis=0)
	events_char_pp_mask = np.broadcast_to(events_char_pp[0,:,:]<1, events_char_pp.shape)
	events_char_pp_xr = xr.DataArray(events_char_pp, name = 'value', coords = [chars,obj_all.lat,obj_all.lon], dims=['char','lat','lon'])
	events_char_pp_xr = events_char_pp_xr.where(~events_char_pp_mask)
	events_char_pp_xr.to_netcdf(dir_comps+'CESM2-LE_'+ens_string+'_dist_mapped_'+hist_var+'_p'+pxx_string+'_FWI_events_char_composites_'+period_str[pp]+'_'+pxx_type+'_'+reg_groups[gg]+'_'+season+'.nc')
	print('\n')
