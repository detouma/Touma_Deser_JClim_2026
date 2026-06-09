#Libraries
import numpy as np
import sys
import glob
import os
import xarray as xr
import FWIsystem
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# set ens in bash script
ens = int(sys.argv[1])
print('ens_ind ='+str(ens), flush=True)

var_ind = int(sys.argv[2])
fwi_vars = ['PRECT','TREFHTMX','RHREFHT','U10']
hist_var_select = np.zeros(len(fwi_vars), dtype='int')
hist_var_select[var_ind] = 1
hist_var = fwi_vars[var_ind]
print(hist_var,flush=True)

hist_type = 'distribution'
pxx_year = 1980

year0 = 1980
year1 = 2082
years = range(year0,(year1+1),1)
nyears = len(years)
ndays = 365

n_ens_dist = 100

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
  fname_dict[fwi_vars[vv]] = 'CESM2-LE_*_'+hist_var+'_mapped_to_'+str(pxx_year)+'_'+str(n_ens_dist)+'_ens_distribution_'+str(year0)+'-'+str(year1)+'*'
  ens_split_dict[fwi_vars[vv]] = '_'
  ens_split_len_dict[fwi_vars[vv]] = -1

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

xr_pr_sel0 = xr_pr.sel(time=xr_pr.time.dt.year.isin(range(year0,year1+1,1)), lat=slice(0,90), lon=slice(180,360)) 
xr_pr_sel = xr_pr_sel0.sel(time=~xr_pr_sel0.get_index("time").duplicated())

time = xr_pr_sel.time
lat = xr_pr_sel.lat
lon = xr_pr_sel.lon
ntime = len(time)
nlat = len(lat)
nlon = len(lon)

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

xr_tmax_sel0 = xr_tmax.sel(time=xr_tmax.time.dt.year.isin(range(year0,year1+1,1)), lat=slice(0,90), lon=slice(180,360))
xr_tmax_sel = xr_tmax_sel0.sel(time=~xr_tmax_sel0.get_index("time").duplicated())

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

xr_rh_sel0 = xr_rh.sel(time=xr_rh.time.dt.year.isin(range(year0,year1+1,1)), lat=slice(0,90), lon=slice(180,360))
xr_rh_sel = xr_rh_sel0.sel(time=~xr_rh_sel0.get_index("time").duplicated())

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

xr_wind_sel0 = xr_wind.sel(time=xr_wind.time.dt.year.isin(range(year0,year1+1,1)), lat=slice(0,90), lon=slice(180,360))
xr_wind_sel = xr_wind_sel0.sel(time=~xr_wind_sel0.get_index("time").duplicated())

if (len(set([pr_ens_string,tmax_ens_string,rh_ens_string,wind_ens_string])) == 1):
 ens_string = pr_ens_string
 print('ens matched successfully', flush=True)
 print(ens_string, flush=True)
else:
 print('ENS_MISMATCH_ERROR', flush=True)
 print([pr_ens_string,tmax_ens_string,rh_ens_string,wind_ens_string], flush=True)
 sys.exit()

print('creating np arrays from xarrays', flush=True)

print('PR', flush=True)
PR0 = np.array(xr_pr_sel.PRECT)
PR = PR0*86400*1000 #to mm/day

print('TMAX', flush=True)
TMAX0 = np.array(xr_tmax_sel.TREFHTMX)
TMAX = TMAX0 - 273.15 #K to deg C 

print('RH', flush=True)
RH = np.array(xr_rh_sel.RHREFHT)
RH = np.ma.masked_array(RH,mask=(RH>100))

print('WIND', flush=True)
WIND0 = np.array(xr_wind_sel.U10)
WIND = WIND0*3.6 #m/s to km/h

# FFMC Calculation
print('calculating FFMC', flush=True)
m_initial = 125  # initial moisture content - guess
FFMC = FWIsystem.calculateFFMC(PR, TMAX, RH, WIND,
                              ntime, nlat, nlon,
                              m_initial,
                              )
# DMC Calculation
print('calculating DMC', flush=True)
day_length = np.full(fill_value=-999.0,shape=(ndays,nlat,nlon))
for dd in range(0,ndays,1):
 doy = dd + 1
 day_length_1d = FWIsystem.daylength(doy,lat) 
 day_length[dd,:,:] = np.broadcast_to(day_length_1d, shape=(nlon,nlat)).T

L_eff = day_length - 3 # approx 3 hours less than duration of daylight
L_eff[L_eff<0] = 0

dmc_initial = 50

DMC = FWIsystem.calculateDMC(PR, TMAX, RH,
                              ntime, nlat, nlon, ndays,
                              dmc_initial, L_eff,
                              )

# DC Calculation

print('calculating DC', flush=True)
# global implementation of effective day length
L_f = 1.43*L_eff - 4.25
L_f[L_f<-1.6] = -1.6

dc_initial = 15

DC = FWIsystem.calculateDC(PR, TMAX,
                              ntime, nlat, nlon, ndays,
                              dc_initial, L_f,
                              ) 

print('calculating ISI', flush=True)

ISI = FWIsystem.calculateISI(FFMC, WIND,
                             ntime, nlat, nlon,
                             )

print('calculating BUI', flush=True)

BUI = FWIsystem.calculateBUI(DMC, DC,
                             ntime, nlat, nlon,
                             )

print('calculating FWI', flush=True)

FWI = FWIsystem.calculateFWI(BUI, ISI,
                             ntime, nlat, nlon,
                             )
# write files
print('writing FWI file', flush=True)
FWIsystem.write_array(FWI, 'FWI',
                      '/glade/derecho/scratch/detouma/fire-precip/FWI/FWI/',
                      hist_var, hist_type,
                      ens_string, pxx_year, year0, year1,
                      time, lat, lon,
                      )
