import numpy as np
import xarray as xr

def daylength(dayOfYear, lat):
    """
    Calculates length of day for use in DMC and DC calculations.
    copied from https://gist.github.com/anttilipp/ed3ab35258c7636d87de6499475301ce
    """
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0+dayOfYear)/365.0))
    if_calc = -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))
    return_array = np.full(fill_value=-999.0,shape=lat.shape)
    return_array[if_calc<=-1.0] = 24.0
    return_array[if_calc>=1.0] = 0.0
    hour_angle= np.rad2deg(np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))))
    return_array[(if_calc>-1.0)&(if_calc<1.0)] = 2*hour_angle[(if_calc>-1.0)&(if_calc<1.0)]/15
    return return_array

def calculateFFMC(PR, TMAX, RH, WIND, 
              ntime, nlat, nlon, 
              m_initial):
    """
    Calculates FFMC for the FWI system following Dowdy et al. (2009)
    PR, TMAX, RH, and WIND must be numpy arrays (time x lat x lon)
    set m_initial guess outside of function
    Returns FFMC as numpy array
    """
    m0 = np.full(fill_value=m_initial,shape=(nlat,nlon))
    FFMC = np.zeros([ntime,nlat,nlon])
    # Influence of rainfall
    for dd in range(0,ntime,1):
        # effective rainfall
        R_eff = PR[dd,:,:]-0.5
        R_eff[R_eff<0] = 0
        # drying diffusion rate k_d
        k_d = 0.581*np.exp(0.0365*TMAX[dd,:,:])*(0.424*(1-(RH[dd,:,:]/100)**1.7) + 0.0694*np.sqrt(WIND[dd,:,:])*(1-(RH[dd,:,:]/100)**8))
        # drying equilibrium moisture content E_d
        E_d = 0.942*(RH[dd,:,:]**0.679) + 11*np.exp((RH[dd,:,:]-100)/10) + 0.18*(21.1-TMAX[dd,:,:])*(1-np.exp(-0.115*RH[dd,:,:]))
        # wetting diffusion rate k_w
        k_w = 0.581*np.exp(0.0365*TMAX[dd,:,:])*(0.424*(1-((100-RH[dd,:,:])/100)**1.7) + 0.0694*np.sqrt(WIND[dd,:,:])*(1-((100-RH[dd,:,:])/100)**8))
        # wetting equilibrium moisture content E_w
        E_w = 0.618*(RH[dd,:,:]**0.753) + 10*np.exp((RH[dd,:,:]-100)/10) + 0.18*(21.1-TMAX[dd,:,:])*(1-np.exp(-0.115*RH[dd,:,:]))
        m_r = np.full(fill_value=-999.0,shape=m0.shape)
        m_r_small = m0 + (42.5*R_eff*np.exp(-100/(251-m0))*(1-np.exp(-(6.93/R_eff))))
        m_r_large = m_r_small + (0.0015*(m0-150)**2)*np.sqrt(R_eff)
        m_r[m0<=150] = m_r_small[m0<=150]
        m_r[m0>150] = m_r_large[m0>150]
        m_r[m_r>250] = 250
        m_r[m_r<0] = 0
        #
        m = np.full(fill_value=-999.0,shape=m0.shape)
        # if m_r > E_d then fuel is drying at the diffustion rate k_d
        m_drying = E_d + (m_r - E_d)*(10**(-1*k_d))
        # if m_r < E_w then fuel is wetting at the diffustion rate k_w
        m_wetting = E_w - (E_w - m_r)*(10**(-1*k_w))
        #
        m[m_r>E_d] = m_drying[m_r>E_d]
        m[m_r<E_w] = m_wetting[m_r<E_w]
        # if E_w < m_r < E_d then m = m_r
        m[(m_r<E_d)&(m_r>E_w)] = m_r[(m_r<E_d)&(m_r>E_w)]
        # calculate new ffmc
        FFMC[dd,:,:] = 59.5 * (250-m) / (147.2 + m)
        # new moisture content
        m0 = 147.2 * (101 - FFMC[dd,:,:]) / (59.5 + FFMC[dd,:,:])
    return FFMC

def calculateDMC(PR, TMAX, RH, 
              ntime, nlat, nlon, ndays,
              dmc_initial, L_eff):
    """
    Calculates DMC for the FWI system following Dowdy et al. (2009)
    PR, TMAX, and RH must be numpy arrays (time x lat x lon)
    set dmc_initial and calculate L_eff using dayLength outside of function
    Returns DMC as numpy array    
    """
    dmc0 = np.full(fill_value=dmc_initial,shape=(nlat,nlon))
    m0 = 20 + np.exp(5.6348-dmc0/43.43)
    DMC = np.full(fill_value=-999.0,shape=(ntime,nlat,nlon)) 
    for tt in range(0,ntime,1):
        doy = tt%ndays
        # calculate effective rainfall
        R_day = PR[tt,:,:]
        R_eff = 0.92*R_day - 1.27
        R_eff[R_day<=1.5] = 0
        # coefficient b is a function of dmc0
        b = np.full(fill_value=-999.0, shape=m0.shape)
        b_opt1 =  100/(0.5+0.3*dmc0)
        b_opt2 =  14-1.3*np.log(dmc0)
        b_opt3 = 6.2*np.log(dmc0)-17.2
        b[dmc0<=33] = b_opt1[dmc0<=33]
        b[(dmc0>33)&(dmc0<=65)] = b_opt2[(dmc0>33)&(dmc0<=65)]
        b[dmc0>65] = b_opt3[dmc0>65]
        #modified mositure content
        m_r = m0 + (1000*R_eff)/(48.77+b*R_eff)
        # Wetting phase of DMC
        dmc_r = 244.72 - 43.43*np.log(m_r - 20)
        dmc_r[dmc_r<=0] = 0
        # Drying phase of DMC
        dmc_d = 1.894*(TMAX[tt,:,:]+1.1)*(100-RH[tt,:,:])*L_eff[doy,:,:]*10**(-4)
        dmc_d[TMAX[tt,:,:]<-1.1] = 0 # "adjust" for TMAX < -1.1
        # new dmc
        DMC[tt,:,:] = dmc_r + dmc_d
        # moisture content for duff layer
        m = 20 + np.exp(5.6348-DMC[tt,:,:]/43.43)
        # update initial moisture content
        m0 = m
        # update initial dmc
        dmc0 = DMC[tt,:,:]
    return DMC

def calculateDC(PR, TMAX,
              ntime, nlat, nlon, ndays,
              dc_initial, L_f):
    """
    Calculates DC for the FWI system following Dowdy et al. (2009)
    PR and TMAX must be numpy arrays (time x lat x lon)
    set dc_initial and calculate L_eff using dayLength and then L_f outside of function
    Returns DC as numpy array    
    """
    dc0 = np.full(fill_value=dc_initial,shape=(nlat,nlon))
    DC = np.full(fill_value=-999.0,shape=(ntime,nlat,nlon))
    for tt in range(0,ntime,1):
        doy = tt%ndays
        #effective rainfall
        R_day = PR[tt,:,:]
        R_eff = 0.83*R_day - 1.27
        R_eff[R_day<=2.8] = 0
        # rainfall phase
        q0 = 800*np.exp((-1*dc0)/400)
        q_r = q0 + 3.937*R_eff
        dc_r = 400*np.log(800/q_r)
        dc_r[dc_r<=0] = 0
        # drying phase
        tmax_day = TMAX[tt,:,:]
        L_f_day = L_f[doy,:,:]
        dc_d = 0.5*(0.36*(tmax_day+2.8)+L_f_day)
        dc_d[tmax_day<=-2.8] = 0.5*L_f_day[tmax_day<=-2.8]
        # new drought code
        DC[tt,:,:] = dc_r+dc_d
        # update initial values
        dc0 = DC[tt,:,:]
    return DC

def calculateISI(FFMC, WIND,
                 ntime, nlat, nlon,
                 ):
    """
    Calculates ISI for the FWI system following Dowdy et al. (2009)
    FFMC and WIND must be numpy arrays (time x lat x lon)
    Returns ISI as numpy array    
    """
    WIND = np.ma.masked_array(WIND,mask=(WIND>100)) # limited to 100 km/h
    ISI = np.full(fill_value=-999.0,shape=(ntime,nlat,nlon))
    for tt in range(0,ntime,1):
        FW = np.exp(0.05039*WIND[tt,:,:])
        m = 147.2 * (101.0 - FFMC[tt,:,:]) / (59.5 + FFMC[tt,:,:])
        FF = 91.9*np.exp(-0.1386*m)*(1+((m**5.31)/(4.93*10**7)))
        ISI[tt,:,:] = 0.208*FW*FF
    return ISI

def calculateBUI(DMC, DC,
                 ntime, nlat, nlon,
                 ):
    """
    Calculates BUI for the FWI system following Dowdy et al. (2009)
    DMC and DC must be numpy arrays (time x lat x lon)
    Returns BUI as numpy array    
    """
    BUI = np.full(fill_value=-999.0,shape=(ntime,nlat,nlon))
    for tt in range(0,ntime,1):
        DMC_day = DMC[tt,:,:]
        DC_day = DC[tt,:,:]
        bui_opt1 = 0.8*DMC_day*DC_day/(DMC_day+0.4*DC_day)
        bui_opt2 = DMC_day - (1 - 0.8*(DMC_day+0.4*DC_day)) * (0.92 + (0.0114*DMC_day)**1.7)
        BUI_day = np.full(fill_value=-999.0,shape=DMC_day.shape)
        BUI_day[DMC_day<=(0.4*DC_day)] = bui_opt1[DMC_day<=(0.4*DC_day)]
        BUI_day[DMC_day>(0.4*DC_day)] = bui_opt2[DMC_day>(0.4*DC_day)]
        BUI_day[BUI_day<0] = 0
        BUI[tt,:,:] = BUI_day
    return BUI

def calculateFWI(BUI, ISI,
                 ntime, nlat, nlon,
                 ):
    """
    Calculates FWI for the FWI system following Dowdy et al. (2009)
    BUI and ISI must be numpy arrays (time x lat x lon)
    Returns FWI as numpy array    
    """
    FWI = np.full(fill_value=-999.0,shape=(ntime, nlat, nlon))
    for tt in range(0,ntime,1):
        BUI_day = BUI[tt,:,:]
        ISI_day = ISI[tt,:,:]
        Fd_opt1 = 0.626*BUI_day**0.809 + 1
        Fd_opt2 = 1000/(25 + 108.64*np.exp(-0.023*BUI_day))
        Fd = np.full(fill_value=-999.0,shape=BUI_day.shape)
        Fd[BUI_day<=80] = Fd_opt1[BUI_day<=80]
        Fd[BUI_day>80] = Fd_opt2[BUI_day>80]
        B = 0.1*Fd*ISI_day
        FWI_day = B
        FWI_day[B>=1] = np.exp(2.72*(0.434*np.log(B[B>=1]))**0.647)
        FWI[tt,:,:] = FWI_day
    return FWI

def write_array(array, fwi_name: str, dir_out: str,
                hist_var: str, hist_type: str,
               ens_string: str, pxx_year, year0, year1,
               time, lat, lon):
    """
    Writes any of the FWI codes to a netcdf file.
    pass the array as a numpy array (direct output of the FWI system calculation functions)
    """
    xr_array = xr.DataArray(array, name = fwi_name, dims=['time','lat','lon'], coords = dict(time=time, lat=lat, lon=lon))
    xr_array.attrs['description'] = fwi_name + ' calculated using Dowdy et al. (2009) - scripts by D. Touma (2025)'
    nc_encoding = {fwi_name: {'_FillValue':-999.0}}
    if (hist_var and hist_type):
        xr_array.attrs['hist_var'] = hist_var
        xr_array.to_netcdf(dir_out+fwi_name+'_CESM2-LE_'+ens_string+'_historic_'+str(pxx_year)+'_'+hist_type+'_mapped_'+hist_var+'_'+str(year0)+'0101-'+str(year1)+'1231.nc', encoding=nc_encoding) 
    else:
        xr_array.to_netcdf(dir_out+fwi_name+'_CESM2-LE_'+ens_string+'_'+str(year0)+'0101-'+str(year1)+'1231.nc', encoding=nc_encoding) 