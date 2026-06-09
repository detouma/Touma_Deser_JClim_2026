import numpy as np
import sys
import pandas as pd
import xarray as xr
import glob
import os
from datetime import datetime

dir_obj = '/glade/campaign/cgd/cas/detouma/fwi_objects/3d_moving_objects/'
dir_tables = '/glade/campaign/cgd/cas/detouma/fwi_objects/object_tables/grp_tables/moving/'
dir_vars = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/'
dir_var_anoms = '/glade/campaign/cgd/cas/detouma/FWI_vars/'
dir_comps = '/glade/derecho/scratch/detouma/extreme_objects/composites/'
dir_fwi = '/glade/campaign/cgd/cas/detouma/FWI_CESM2LE/'

nn = int(sys.argv[1])
gg = int(sys.argv[2])

print('nn = '+str(nn)+', gg = '+str(gg), flush=True)

pxx_year0 = 1980
pxx_year1 = 2014
nyears_window = 35
pxx = 99.0
pxx_string =  str(pxx)

reg_groups = ['PacCoast','FourCorners']

year0 = 1980
year1 = 2082 #2100
years = np.arange(year0,(year1+1),1)
nyears = len(years)

season_months = [6, 7, 8, 9, 10]
season = 'JJASO'

pyear0 = [1980,2015,2050]
pyear1 = [2014,2050,2082]
nyears_period = np.array(pyear1)-np.array(pyear0)+1
nperiods = len(pyear0)
pxx_year0 = 1980
pxx_year1 = 2014
period_str = ['-'.join(i) for i in zip([str(i) for i in pyear0],[str(i) for i in pyear1])]

files_3d_list = []

for file in sorted(glob.glob(dir_tables+'CESM2-LE_*_FWI_allens_moving_p'+pxx_string+'_3d_events_'+str(year0)+'-'+str(year1)+'*.txt')):
    files_3d_list.append(os.path.relpath(file, dir_tables))

nfiles_3d = len(files_3d_list)

ens_3d = np.array([i.split('_')[1] for i in files_3d_list])
reg_3d_0 =  [i.split('_')[9] for i in files_3d_list]
reg_3d =  np.array([i.split('.')[0] for i in reg_3d_0])
files_df = pd.DataFrame({'ens':ens_3d,'reg':reg_3d,'filename':files_3d_list})

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
print(all_events_3d.month[0:10], flush=True)

chars = ['nevents', 'ndays', 'area' ,'max_fwi', 'avg_fwi', 'pr_anom', 'tmax_anom', 'rh_anom', 'wind_anom', 'pr_mov_anom', 'tmax_mov_anom', 'rh_mov_anom', 'wind_mov_anom', 'pr', 'tmax', 'rh', 'wind']

# read in FWI
print('read in FWI', flush=True)
f_fwi = glob.glob(dir_fwi+'*'+ens_string+'*.nc')[0]
xr_fwi = xr.open_dataset(f_fwi)
fwi = xr_fwi.FWI.sel(date=xr_fwi.date.isin(range(year0*10000,(year1+1)*10000,1)),lon=slice(180,360), lat=slice(0,90))

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
pr_files_ens_inds = np.where(np.array(pr_files_ens_str)==ens_string)[0]
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
tmax_files_ens_inds = np.where(np.array(tmax_files_ens_str)==ens_string)[0]
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
rh_files_ens_inds = np.where(np.array(rh_files_ens_str)==ens_string)[0]
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
wind_files_ens_inds = np.where(np.array(wind_files_ens_str)==ens_string)[0]
wind_files_ens = []
for ii in wind_files_ens_inds:
 wind_files_ens.append(wind_files[ii])

xr_wind = xr.open_mfdataset([dir_vars+'U10/'+x for x in wind_files_ens], preprocess=xr_coord_sel)
xr_wind_time = xr_wind.sel(time=xr_wind.time.dt.year.isin(range(year0,year1+1,1)))
wind = np.array(xr_wind_time.U10.sel(time=~xr_wind_time.get_index("time").duplicated()))
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
# pr_mov_mean = xr.DataArray(pr_mov_mean_doy_np,name='PRECT', coords=xr_pr.coords, dims=xr_pr.dims)
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

print('reading moving object file', flush=True)
moving_object_file = 'CESM2-LE_'+ens_string+'_allens_moving_p'+pxx_string+'_FWI_3d_event_numbers_'+str(year0)+'-'+str(year1)+'_'+reg_groups[gg]+'.nc'
xr_moving_obj = xr.open_dataset(dir_obj+moving_object_file)
nlat = len(xr_moving_obj.lat)
nlon = len(xr_moving_obj.lon)
obj_pxx = xr_moving_obj.event 

for pp in range(0,nperiods,1):
	print(reg_groups[gg] + ' | ' + period_str[pp], flush=True)
	event_subset_df = all_events_3d[(all_events_3d['region']==reg_groups[gg])&(all_events_3d['month'].isin(season_months))&(all_events_3d['period']==period_str[pp])].reset_index()
	n_events_subset = len(event_subset_df)
	events_char_pp_xx = np.full(fill_value = -999.0, shape=(len(chars),nlat,nlon))
	pp_xx_event_mask = np.zeros(obj_pxx.shape,dtype=int) #time x lat x lon - where events occur in time and space
	pp_xx_nevents = np.zeros((nlat,nlon),dtype=int) #lat x lon - add up numper of events for each grid point
	pp_xx_events_ndays = np.zeros((n_events_subset,nlat,nlon),dtype=int) #events x lat x lon - number of days for each event at each grid point
	pp_xx_events_area = np.zeros((n_events_subset,nlat,nlon),dtype=int) # events x lat x lon - area of each event at each grod point
	for nn in range(0,n_events_subset,1):
		if nn%5==0:
			print(str(nn)+' out of '+str(n_events_subset), end= ', ', flush=True)
		event_id_nn = event_subset_df['event_id'][nn]
		nn_inds = np.where(obj_pxx==event_id_nn)
		pp_xx_event_mask[nn_inds] = 1
		pp_xx_nevents[nn_inds[1],nn_inds[2]] += 1
		pp_xx_events_ndays[nn,nn_inds[1],nn_inds[2]] = event_subset_df['ndays'][nn]
		pp_xx_events_area[nn,nn_inds[1],nn_inds[2]] = event_subset_df['area_3d'][nn] 
	print('nevents', flush=True) 
	events_char_pp_xx[0,:,:] = pp_xx_nevents
	print('ndays', flush=True) 
	pp_xx_events_ndays = np.ma.masked_array(pp_xx_events_ndays, mask=(pp_xx_events_ndays==0))
	events_char_pp_xx[1,:,:] = np.ma.sum(pp_xx_events_ndays,axis=0) # ndays
	print('area', flush=True) 
	pp_xx_events_area = np.ma.masked_array(pp_xx_events_area, mask=(pp_xx_events_area==0))
	events_char_pp_xx[2,:,:] = np.ma.mean(pp_xx_events_area,axis=0) #area
	print('max_fwi', flush=True) 
	events_char_pp_xx[3,:,:] = np.ma.max(np.ma.masked_array(fwi,mask=(pp_xx_event_mask==0)),axis=0)
	print('avg_fwi', flush=True) 
	events_char_pp_xx[4,:,:] = np.ma.mean(np.ma.masked_array(fwi,mask=(pp_xx_event_mask==0)),axis=0)
	for cc in range(5,len(chars),1):
		print(chars[cc], flush=True) 
		events_char_pp_xx[cc,:,:] = np.ma.mean(np.ma.masked_array(eval(chars[cc]),mask=(pp_xx_event_mask==0)),axis=0)
	events_char_pp_xx_mask = np.broadcast_to(events_char_pp_xx[0,:,:]<1, events_char_pp_xx.shape)
	events_char_pp_xx_xr = xr.DataArray(events_char_pp_xx, name = 'value', coords = [chars,obj_pxx.lat,obj_pxx.lon], dims=['char','lat','lon'])
	events_char_pp_xx_xr = events_char_pp_xx_xr.where(~events_char_pp_xx_mask)
	events_char_pp_xx_xr.to_netcdf(dir_comps+'CESM2-LE_'+ens_string+'_p'+pxx_string+'_FWI_events_char_composites_'+period_str[pp]+'_moving_'+reg_groups[gg]+'_'+season+'.nc')
	print('\n')
