import numpy as np
import sys
import os
import glob
import xarray as xr

# argument inputs
ens_subset_len = int(sys.argv[1])

var_abbr = 'VP'
var_name = 'VP'

print(var_name, flush=True)
dir_anom = '/glade/derecho/scratch/detouma/extreme_objects/'+var_name+'/'

# constant inputs
ndays_window = 30
nyears_window = 34

year0 = 1980
year1 = 2082
years = np.arange(year0,year1+1,1)

pxx_year = 1980

# finding all files for variable
files_list = []
for file in sorted(glob.glob(dir_anom+'CESM2-LE_'+str(ens_subset_len)+'_ens_members_'+var_name+'_mapped_to_'+str(pxx_year)+'_distribution_doy_*_'+str(year0)+'-'+str(year1)+'.nc')):
        files_list.append(os.path.relpath(file, dir_anom))

file_doys = np.array([i.split('_')[-2] for i in files_list]).astype(int)

check_for_all_files = np.all(np.isin(np.arange(1,366,1),file_doys))

if ~check_for_all_files:
	print('not all files present', flush=True)
	missing_doys = np.where(~np.isin(np.arange(1,366,1),file_doys))[0]+1
	print('missing files: ', end='', flush = True)
	for dd in missing_doys:
		print(dd, end=', ')
	print('exiting python')
	exit()
else:
	print('all files present')

# find ens member strings
xr_doy1 = xr.open_dataset(dir_anom+files_list[0])
ens_members = xr_doy1.ens_member.values

for ens in ens_members:
	print(ens, end=', ', flush=True)
	xr_ens_list = []
	for f in files_list:
		xr_doy = xr.open_dataset(dir_anom+f)
		xr_ens_list.append(xr_doy.sel(ens_member=ens))
	xr_ens_unsorted = xr.concat(xr_ens_list, dim='time')
	xr_ens = xr_ens_unsorted.sortby(xr_ens_unsorted.time)
	xr_ens.to_netcdf(dir_anom+'CESM2-LE_'+ens+'_'+var_name+'_mapped_to_'+str(pxx_year)+'_'+str(ens_subset_len)+'_ens_distribution_'+str(year0)+'-'+str(year1)+'.nc')

