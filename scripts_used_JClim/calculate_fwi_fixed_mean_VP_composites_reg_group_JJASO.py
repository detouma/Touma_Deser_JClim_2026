import numpy as np
import sys
import pandas as pd
import xarray as xr
import glob
import os
from datetime import datetime

dir_obj = "/glade/campaign/cgd/cas/detouma/fwi_objects/3d_fixed_objects/"
dir_tables = '/glade/campaign/cgd/cas/detouma/fwi_objects/object_tables/grp_tables/fixed/'
dir_vars = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/'
dir_var_anoms = '/glade/campaign/cgd/cas/detouma/FWI_vars/'
dir_comps = '/glade/derecho/scratch/detouma/extreme_objects/composites/'
dir_vp = '/glade/derecho/scratch/detouma/extreme_objects/VP/'
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
# seasons = ['DJF','MAM','JJA','SON']
# nseasons = len(seasons)

pyear0 = [1980,2015,2050]
pyear1 = [2014,2050,2082]
nyears_period = np.array(pyear1)-np.array(pyear0)+1
nperiods = len(pyear0)
pxx_year0 = 1980
pxx_year1 = 2014
period_str = ['-'.join(i) for i in zip([str(i) for i in pyear0],[str(i) for i in pyear1])]

files_3d_list = []

for file in sorted(glob.glob(dir_tables+'CESM2-LE_*_FWI_allens_*_p'+pxx_string+'_3d_events_'+str(year0)+'-'+str(year1)+'_*.txt')):
    files_3d_list.append(os.path.relpath(file, dir_tables))

nfiles_3d = len(files_3d_list)

ens_3d = np.array([i.split('_')[1] for i in files_3d_list])
reg_3d_0 =  [i.split('_')[9] for i in files_3d_list]
reg_3d =  np.array([i.split('.')[0] for i in reg_3d_0])
files_df = pd.DataFrame({'ens':ens_3d,
                         'reg':reg_3d,
                         'filename':files_3d_list})

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

# chars = ['nevents', 'ndays', 'area' ,'max_fwi', 'avg_fwi', 'pr_anom', 'tmax_anom', 'rh_anom', 'wind_anom', 'pr_mov_anom', 'tmax_mov_anom', 'rh_mov_anom', 'wind_mov_anom', 'pr', 'tmax', 'rh', 'wind']
chars = ['nevents', 'ndays', 'area', 'vapor_pressure_anom', 'vapor_pressure_mov_anom']

# read in VP
print('read in VP', flush=True)
f_vp = glob.glob(dir_vp+'*'+ens_string+'*.nc')[0]
xr_vp = xr.open_dataset(f_vp)
vp = xr_vp.VP.sel(time=xr_vp.time.dt.year.isin(range(year0,year1+1,1)))

#preprocessing function for reading multiyear files
def preprocess(ds):
    return ds.expand_dims(year = [datetime.now()])

#calculate vp mov anomaly
print('calculate vp mov anomaly', flush=True)
vp_mov_mean_files_list = []

for yy in years:
    vp_mov_mean_files_list.append(dir_vp+'mean/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_VapPress_'+str(year0)+'_NWHemi.nc')

vp_mov_mean_xr = xr.open_mfdataset(vp_mov_mean_files_list, preprocess=preprocess)
vp_mov_mean_xr['year'] =  years
vp_mov_mean_doy = vp_mov_mean_xr.VP
vp_mov_mean = np.reshape(np.array(vp_mov_mean_doy),newshape=((nyears*365),len(vp_mov_mean_doy.lat),len(vp_mov_mean_doy.lon)))
vp_mov_anom = vp - vp_mov_mean

#calculate vp anomaly
print('calculate vp anomaly', flush=True)
f_vp_mean = dir_vp+'mean/CESM2-LE_allens_mean_30day_mov_34year_mov_avg_VapPress_'+str(year0)+'_NWHemi.nc'
xr_vp_mean = xr.open_dataset(f_vp_mean)
vp_mean_365 = np.array(xr_vp_mean.VP) 
vp_mean = np.tile(vp_mean_365,[nyears,1,1])
vp_anom = vp-vp_mean

# object files
print('reading fixed object file', flush=True)
fixed_object_file = 'CESM2-LE_'+ens_string+'_allens_fixed_p'+pxx_string+'_FWI_3d_event_numbers_'+str(year0)+'-'+str(year1)+'_'+reg_groups[gg]+'.nc'
xr_fixed_obj = xr.open_dataset(dir_obj+fixed_object_file)
nlat = len(xr_fixed_obj.lat)
nlon = len(xr_fixed_obj.lon)

fixed_obj = xr_fixed_obj.event 


for pp in range(0,nperiods,1):
	print(reg_groups[gg] + ' | ' + season+ ' | ' + period_str[pp], flush=True)
	event_subset_df = all_events_3d[(all_events_3d['region']==reg_groups[gg])&(all_events_3d['month'].isin(season_months))&(all_events_3d['period']==period_str[pp])].reset_index()
	n_events_subset = len(event_subset_df)
	events_char_pp_xx = np.full(fill_value = -999.0, shape=(len(chars),nlat,nlon))
	pp_xx_event_mask = np.zeros(fixed_obj.shape,dtype=int) #time x lat x lon - where events occur in time and space
	pp_xx_nevents = np.zeros((nlat,nlon),dtype=int) #lat x lon - add up numper of events for each grid point
	pp_xx_events_ndays = np.zeros((n_events_subset,nlat,nlon),dtype=int) #events x lat x lon - number of days for each event at each grid point
	pp_xx_events_area = np.zeros((n_events_subset,nlat,nlon),dtype=int) # events x lat x lon - area of each event at each grod point
	for nn in range(0,n_events_subset,1):
		if nn%5==0:
			print(str(nn)+' out of '+str(n_events_subset), end= ', ', flush=True)
		event_id_nn = event_subset_df['event_id'][nn]
		nn_inds = np.where(fixed_obj==event_id_nn)
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
	print('vp_anom', flush=True) 
	events_char_pp_xx[3,:,:] = np.ma.mean(np.ma.masked_array(vp_anom,mask=(pp_xx_event_mask==0)),axis=0)
	print('vp_mov_anom', flush=True) 
	events_char_pp_xx[4,:,:] = np.ma.mean(np.ma.masked_array(vp_mov_anom,mask=(pp_xx_event_mask==0)),axis=0)
	events_char_pp_xx_mask = np.broadcast_to(events_char_pp_xx[0,:,:]<1, events_char_pp_xx.shape)
	events_char_pp_xx_xr = xr.DataArray(events_char_pp_xx, name = 'value', coords = [chars,xr_fixed_obj.lat,xr_fixed_obj.lon], dims=['char','lat','lon'])
	events_char_pp_xx_xr = events_char_pp_xx_xr.where(~events_char_pp_xx_mask)
	events_char_pp_xx_xr.to_netcdf(dir_comps+'CESM2-LE_'+ens_string+'_p'+pxx_string+'_FWI_events_char_VP_composites_'+period_str[pp]+'_fixed_'+reg_groups[gg]+'_'+season+'.nc')
	print('\n')
