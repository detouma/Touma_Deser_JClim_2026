import numpy as np
import sys
import os
import glob
import xarray as xr
import time
from netCDF4 import Dataset

script_start = time.time()
# argument inputs
doy = int(sys.argv[1]) # from 1 to 365
ens_subset_len0 = int(sys.argv[2]) # how many ensemble members to use (tested with 20)

var_name = 'VP'

# constant inputs
ndays_window = 30
nyears_window = 34
ndays = 365

year0 = 1980
year1 = 2082
nyears = year1-year0+1

pxx_year = 1980
pxx_year0 = int(pxx_year-nyears_window/2)
pxx_year1 = int(pxx_year+nyears_window/2)

pxx_years = range(pxx_year0,(pxx_year1+1),1)
pxx_nyears = len(pxx_years)

pxx_ntime = pxx_nyears*ndays

print('doy = '+str(doy), end = ' | ', flush=True)
print('var = '+ var_name, end = ' | ', flush=True)
print('ens subet len = '+str(ens_subset_len0), flush=True)

# directories
dir_var = '/glade/campaign/cgd/cas/detouma/FWI_vars/VP/'
dir_anom = '/glade/derecho/scratch/detouma/extreme_objects/'+var_name+'/'

# finding files for variable and nyear window
files_list = []
for file in sorted(glob.glob(dir_var+'*.nc')):
	files_list.append(os.path.relpath(file, dir_var))

ens_string = [i.split('_')[2] for i in files_list]

xr_ens0 = xr.open_dataset(dir_var+files_list[0])
lat0 = xr_ens0.lat
lon0 = xr_ens0.lon
lat_inds = np.where(lat0>=0)[0]
lon_inds = np.where(lon0>=180)[0]
lat = lat0[lat_inds]
lon = lon0[lon_inds]
nlat = len(lat)
nlon = len(lon)

xr_ens0_units = xr_ens0[var_name].units 

ens_subset_len = np.min([len(ens_string),ens_subset_len0])

# find doy range
if (doy<=(ndays_window/2)):
 tt0 = int(ndays+doy-(ndays_window/2))
 doy_range = np.append(np.arange(tt0,ndays+1,1),np.arange(1,ndays_window-(ndays-tt0),1))
elif (doy+(ndays_window/2)>(ndays+1)):
 tt1 = int(doy+ndays_window/2-ndays)
 doy_range = np.append(np.arange(doy-int(ndays_window/2),ndays+1,1),np.arange(1,tt1,1))
else:
 doy_range = np.arange(doy-int(ndays_window/2),doy+int(ndays_window/2),1)

var_mid_doy = np.zeros((ens_subset_len,nyears,nlat,nlon))
var_doy = np.zeros((ens_subset_len,ndays_window*(nyears+nyears_window)+1, nlat, nlon)) # add 1 because of duplicate time
hist_doy = np.zeros((ens_subset_len,ndays_window*pxx_nyears, nlat, nlon))

var_mid_doy_time_list = []
var_doy_time_list = []
hist_doy_time_list = []

# reading in ens subset only
print('reading in '+str(ens_subset_len)+' ens subset '+str(year0)+'-'+str(year1)+' | '+var_name ,flush = True)
# var_list = []
for nn in range(0,ens_subset_len,1):
	ens_loop_start = time.time()
	ens_nn = ens_string[nn]
	print(nn, end=': ', flush=True)
	tt_doy_count = 0
	tt_hist_count = 0
	xr_ens = xr.open_dataset(dir_var+files_list[nn])
	nc_ens = Dataset(dir_var+files_list[nn], 'r')
	ens_time = xr_ens.time
	#####
	ens_ff_var_doy_inds = np.where((np.isin(np.array(ens_time.dt.year),range(year0-int(nyears_window/2),year1+int(nyears_window/2)+1,1)))
								&(np.isin(np.array(ens_time.dt.dayofyear),doy_range))
								&(~ens_time.get_index("time").duplicated())
									)[0]
	if len(ens_ff_var_doy_inds>0):
		var_doy[nn,tt_doy_count:tt_doy_count+len(ens_ff_var_doy_inds),:,:] = nc_ens[var_name][ens_ff_var_doy_inds,lat_inds,lon_inds]
		tt_doy_count += len(ens_ff_var_doy_inds)
		if nn==0:
			var_doy_time_list.append(ens_time[ens_ff_var_doy_inds])
	#####
	ens_ff_hist_doy_inds = np.where((np.isin(np.array(ens_time.dt.year),pxx_years))
								   &(np.isin(np.array(ens_time.dt.dayofyear),doy_range))
								   &(~ens_time.get_index("time").duplicated())
								)[0]
	if len(ens_ff_hist_doy_inds>0):
		hist_doy[nn,tt_hist_count:tt_hist_count+len(ens_ff_hist_doy_inds),:,:] = nc_ens[var_name][ens_ff_hist_doy_inds,lat_inds,lon_inds]
		tt_hist_count += len(ens_ff_hist_doy_inds)
		if nn==0:
			hist_doy_time_list.append(ens_time[ens_ff_hist_doy_inds])
	#####
	ens_ff_mid_doy_inds = np.where((np.isin(np.array(ens_time.dt.year),range(year0,year1+1,1)))
								   &(np.array(ens_time.dt.dayofyear)==doy)
								   &(~ens_time.get_index("time").duplicated())
								)[0]
	if len(ens_ff_mid_doy_inds>0):
		ens_ff_mid_doy_inds_years = ens_time.dt.year[ens_ff_mid_doy_inds]
		var_mid_doy_year_inds = np.where(np.isin(np.arange(year0,year1+1,1),ens_ff_mid_doy_inds_years))[0]	
		var_mid_doy[nn,var_mid_doy_year_inds,:,:] = nc_ens[var_name][ens_ff_mid_doy_inds,lat_inds,lon_inds]
		if nn==0:
			var_mid_doy_time_list.append(ens_time[ens_ff_mid_doy_inds]) 
	ens_loop_end = time.time()
	ens_loop_time = ens_loop_end-ens_loop_start
	print(str(ens_loop_time)+'s')

var_mid_doy_time = xr.concat(var_mid_doy_time_list, dim='time')
var_doy_time = xr.concat(var_doy_time_list, dim='time')
hist_doy_time = xr.concat(hist_doy_time_list, dim='time') #unused right now

var_mid_doy_time = var_mid_doy_time.drop_duplicates(dim='time')

print('reshaping hist doy array')
hist_doy_all = np.reshape(hist_doy,newshape=(ens_subset_len*ndays_window*pxx_nyears, nlat, nlon))
print('sort hist doy array', end=': ',flush=True)
hist_sort_start = time.time()
hist_doy_sorted = np.sort(hist_doy_all,axis=0)
hist_sort_end = time.time()
hist_sort_time = hist_sort_end-hist_sort_start
print(str(hist_sort_time)+'s')


var_mapped_ens_doy = np.full(fill_value=-999.0,shape=(ens_subset_len,nyears,nlat,nlon))

#trying np.searchsorted
print('start variable mapping loop', flush=True)
loop_start = time.time()
for yy in range(0,nyears,1):
	yy_loop_start = time.time()
	print(year0+yy, end=' | ', flush=True)
	var_doy_year_inds = np.where(var_doy_time.time.dt.year.isin(np.arange(year0+yy-nyears_window/2,year0+yy+nyears_window/2+1,1)))[0] 
	var_doy_year_all = np.reshape(var_doy[:,var_doy_year_inds,:,:], newshape=(ens_subset_len*len(var_doy_year_inds),nlat,nlon)) 
	var_doy_year_all_sorted = np.sort(var_doy_year_all,axis=0) 
	for ii in range(0,nlat,1):
		for jj in range(0,nlon,1):
			ind_ii_jj_yy_left = np.searchsorted(var_doy_year_all_sorted[:,ii,jj],var_mid_doy[:,yy,ii,jj],side='left')
			ind_ii_jj_yy_right = np.searchsorted(var_doy_year_all_sorted[:,ii,jj],var_mid_doy[:,yy,ii,jj],side='right')
			percentile_ii_jj_yy = ((ind_ii_jj_yy_left+ind_ii_jj_yy_right)/2)/var_doy_year_all_sorted.shape[0]
			var_ii_jj_yy_mapped_inds = np.floor(percentile_ii_jj_yy*hist_doy_sorted.shape[0]).astype(int)	
			var_mapped_ens_doy[:,yy,ii,jj] = hist_doy_sorted[var_ii_jj_yy_mapped_inds,ii,jj]
	yy_loop_end = time.time()
	print('time for year: '+str(yy_loop_end-yy_loop_start)+ 's', flush=True)

loop_end = time.time()
loop_time = loop_end-loop_start
print('time for all loops = '+str(loop_time)+'s', flush=True)

print('converting np final array to xr array', flush=True)
var_mapped_ens_doy_xr = xr.DataArray(var_mapped_ens_doy, name=var_name, dims=["ens_member", "time", "lat", "lon"], coords=dict(ens_member=ens_string[:ens_subset_len], time=var_mid_doy_time, lat=lat, lon=lon,), attrs=dict(description=var_name+ ' mapped to '+str(pxx_year)+' distribution', units=xr_ens0_units, members_used=str(ens_subset_len)),)

print('writing final xr array file', flush=True)
write_start = time.time()
encoding = {var_name: {'dtype': 'float32'}}
var_mapped_ens_doy_xr.to_netcdf(dir_anom+'CESM2-LE_'+str(ens_subset_len0)+'_ens_members_'+var_name+'_mapped_to_'+str(pxx_year)+'_distribution_doy_'+str(doy)+'_'+str(year0)+'-'+str(year1)+'.nc', encoding=encoding)
write_end = time.time()
write_time = write_end - write_start
print('time for writing = '+str(write_time)+'s', flush=True)

script_end = time.time()
script_time = script_end-script_start
print('total time of script = '+str(script_time)+'s', flush=True)
