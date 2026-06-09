import numpy as np
import sys
import pandas as pd
import xarray as xr
import glob
import os
from datetime import datetime

dir_fwi = '/glade/derecho/scratch/detouma/fire-precip/FWI/RH_VP_mapped/'
dir_obj = '/glade/derecho/scratch/detouma/extreme_objects/fwi_objects/reg_groups/'
dir_tables = '/glade/derecho/scratch/detouma/extreme_objects/object_tables/'
dir_vars = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/'
dir_comps = '/glade/derecho/scratch/detouma/extreme_objects/composites/'

nn = int(sys.argv[1]) # ensemble
gg = int(sys.argv[2]) #reg group

print('nn = '+str(nn)+', gg = '+str(gg), flush=True)

hist_var = 'RHREFHT' 
print(hist_var,flush=True)

pxx_type = 'fixed'

pxx_year0 = 1980
pxx_year1 = 2014
nyears_window = 35
pxx = 99.0
pxx_string = str(pxx)

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

n_ens_dist = 70

files_3d_list = []

print('CESM2-LE_*_FWI_'+str(n_ens_dist)+'_ens_'+pxx_type+'_p'+pxx_string+'_3d_events_'+str(year0)+'-'+str(year1)+'_*_historic_'+hist_var+'_dist_VP.txt')
for file in sorted(glob.glob(dir_tables+'CESM2-LE_*_FWI_'+str(n_ens_dist)+'_ens_'+pxx_type+'_p'+pxx_string+'_3d_events_'+str(year0)+'-'+str(year1)+'_*_historic_'+hist_var+'_dist_VP.txt')):
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

chars = ['nevents', 'ndays', 'area' ,'max_fwi', 'avg_fwi'] #'pr_anom', 'tmax_anom', 'rh_anom', 'wind_anom', 'pr_mov_anom', 'tmax_mov_anom', 'rh_mov_anom', 'wind_mov_anom', 'pr', 'tmax', 'rh', 'wind']

# read in FWI
f_fwi = 'FWI_CESM2-LE_'+ens_string+'_historic_1980_dist_VP_mapped_RHREFHT_19800101-20821231.nc'
fwi_xr = xr.open_dataset(dir_fwi+f_fwi)
fwi_xr_sel = fwi_xr.sel(time=fwi_xr.time.dt.year.isin(np.arange(year0,year1,1)))
fwi = np.array(fwi_xr_sel.FWI)

year = np.array(fwi_xr_sel.time.dt.year)
month = np.array(fwi_xr_sel.time.dt.month)
day = np.array(fwi_xr_sel.time.dt.day)
date = year*10000+month*100+day

# object files
print('reading object file', flush=True)
object_file = 'CESM2-LE_'+ens_string+'_dist_VP_mapped_'+hist_var+'_to_'+str(n_ens_dist)+'ens_'+str(pxx_year0)+'_p'+pxx_string+'_FWI_event_numbers_'+str(year0)+'-'+str(year1)+'_'+reg_groups[gg]+'.nc'
xr_obj = xr.open_dataset(dir_obj+object_file)
nlat = len(xr_obj.lat)
nlon = len(xr_obj.lon)

obj_all = xr_obj.event

for pp in range(0,nperiods,1):
	print(reg_groups[gg] + ' | ' + period_str[pp] + ' | ' + hist_var, flush=True)
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
	# for cc in range(5,len(chars),1):
	# 	print(chars[cc], flush=True) 
	# 	events_char_pp[cc,:,:] = np.ma.mean(np.ma.masked_array(eval(chars[cc]),mask=(pp_event_mask==0)),axis=0)
	events_char_pp_mask = np.broadcast_to(events_char_pp[0,:,:]<1, events_char_pp.shape)
	events_char_pp_xr = xr.DataArray(events_char_pp, name = 'value', coords = [chars,obj_all.lat,obj_all.lon], dims=['char','lat','lon'])
	events_char_pp_xr = events_char_pp_xr.where(~events_char_pp_mask)
	events_char_pp_xr.to_netcdf(dir_comps+'CESM2-LE_'+ens_string+'_dist_VP_mapped_'+hist_var+'_p'+pxx_string+'_FWI_events_char_composites_'+period_str[pp]+'_'+pxx_type+'_'+reg_groups[gg]+'_'+season+'.nc')
	print('\n')
