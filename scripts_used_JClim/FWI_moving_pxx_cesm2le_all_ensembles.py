import numpy as np
import sys
import os
import xarray as xr
import glob

yy = int(sys.argv[1])

dir_fwi = '/glade/campaign/cgd/cas/detouma/FWI_CESM2LE/'
dir_pxx = '/glade/derecho/scratch/detouma/fire-precip/FWI/FWI_pxx_CESM2LE/'

year0 = 1980
year1 = 2100

mid_year = yy+1980

nyears_window = 34
pxx_year0 = int(yy+1980-(nyears_window/2))
pxx_year1 = int(yy+1980+(nyears_window/2))
if pxx_year1>year1:
 pxx_year1=year1
 
print(mid_year, flush=True)
print(str(pxx_year0)+'-'+str(pxx_year1),flush=True)

pxx_list = np.array([90,95,98,99,99.9])
qxx_list = pxx_list/100

files_list = []
for file in glob.glob(dir_fwi+'*U10.nc'):
 files_list.append(os.path.relpath(file, dir_fwi))

fwi_ens_list = []
print('reading in files', flush=True)
for nn in files_list:
    print(nn, end=', ', flush=True)
    file_ens = nn
    xr_ens = xr.open_dataset(dir_fwi+nn,
                            )
    fwi_ens_list.append(xr_ens.FWI.sel(lat=slice(0,90), lon=slice(180,360), 
                        date=slice(pxx_year0*10000,(pxx_year1+1)*10000-1)))

print('concat files', flush=True)
fwi_all = xr.concat(fwi_ens_list, dim='ens_member',coords='minimal')
fwi_pxx =  fwi_all.quantile(qxx_list, dim=["date", "ens_member"])

file_pxx = dir_pxx+'CESM2-LE_FWI_'+str(mid_year)+'_'+str(nyears_window+1)+'-year_moving_high_percentiles_allens_U10_NWHemi.nc'
fwi_pxx.to_netcdf(file_pxx)
