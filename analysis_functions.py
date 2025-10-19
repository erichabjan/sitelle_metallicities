import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib
from matplotlib.colors import LogNorm

from astropy.table import Table, join, QTable, vstack
import astropy.units as u
import sys
import pyneb as pn
from multiprocessing import Pool
import multiprocessing as mp
import math
from astropy.io import fits
from orcs.process import SpectralCube
import orcs
import orb

from astropy.nddata import NDData, Cutout2D
from astropy.wcs import WCS, find_all_wcs
from astropy.wcs.utils import fit_wcs_from_points, pixel_to_skycoord, skycoord_to_pixel

import astropy
import astropy.units as u
from astropy.coordinates import SkyCoord

from reproject import reproject_interp
from reproject import reproject_exact
from reproject import reproject_adaptive
import reproject
from regions import PixCoord

import pylab as pl

from scipy.optimize import curve_fit
from scipy.integrate import quad

from orb.fit import fit_lines_in_spectrum
from orb.utils.spectrum import corr2theta, amp_ratio_from_flux_ratio
from orb.core import Lines
import gvar
import orb

import extinction
from extinction import apply, remove

from photutils.detection import DAOStarFinder

import aplpy

from astropy.stats import sigma_clipped_stats
from photutils.psf import extract_stars
from astropy.visualization import simple_norm
from photutils.psf import EPSFBuilder

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from MontagePy.main import mGetHdr, mProject, mProjectCube



### Sky Backfround subtraction

def skyback(incube, phdata, ingal, cube2d_wcs, in_coords, galaxy_v, mc=False):

    x_out, y_out = cube2d_wcs.world_to_pixel(SkyCoord(in_coords[0], in_coords[1], unit='deg'))
    x, y = int(x_out), int(y_out)

    reg_coord = SkyCoord(phdata[np.max(phdata['deproj_dist']) == phdata['deproj_dist']]['cen_ra'][0], phdata[np.max(phdata['deproj_dist']) == phdata['deproj_dist']]['cen_dec'][0], unit='deg')

    x_max, y_max = cube2d_wcs.world_to_pixel(SkyCoord(reg_coord, unit='deg'))
    x_max, y_max = int(x_max), int(y_max)

    r = np.max([abs(x_max - x), abs(y_max - y)]) #np.max([abs(x_max - x), abs(y_max - y)]) 
    if ingal == 'NGC0628':
        rmin = r * 2.5
    if ingal == 'NGC2835':
        rmin = r * 3
    if ingal == 'NGC3351':
        rmin = r * 3.3
    if ingal == 'NGC4535':
        rmin = r * 3.1
    if ingal == 'NGC3627':
        rmin = r * 2.1
    rmax = np.sqrt(r**2 + rmin**2)

    imin = x - rmax
    imax = x + rmax + 1
    jmin = y - rmax
    jmax = y + rmax + 1
    xlist = []
    ylist = []
    buffval = 150

    for i in np.arange(imin, imax):
        if 0 + buffval <= i <= 2048 - buffval:
            for j in np.arange(jmin, jmax):
                if 0 + buffval <= j <= 2048 - buffval:
                    ij = np.array([i,j])
                    dist = np.linalg.norm(ij - np.array((x,y)))
                    i, j = int(i), int(j)
                    distnum = 3
                    if dist > rmin and dist <= rmax:
                        xlist.append(i)
                        ylist.append(j)

    inpix = (xlist, ylist)
    outspec = np.nanmedian(incube[:, xlist, ylist], axis=1)
    
    if mc == True:
        
        mc = 2
        mc_spec = []
            
        for k in range(mc):
            
            if ingal == 'NGC0628':
                rmin = r * 2.5
            if ingal == 'NGC2835':
                rmin = r * 3
            if ingal == 'NGC3351':
                rmin = r * 3.3
            if ingal == 'NGC4535':
                rmin = r * 3.1
            if ingal == 'NGC3627':
                rmin = r * 2.1
            
            rmin = rmin + np.random.normal(0, 0.5)
            
            rmax = np.sqrt(r**2 + rmin**2)

            imin = x - rmax
            imax = x + rmax + 1
            jmin = y - rmax
            jmax = y + rmax + 1
            xlist = []
            ylist = []
            buffval = 50

            for i in np.arange(imin, imax):
                if 0 + buffval <= i <= 2048 - buffval:
                    for j in np.arange(jmin, jmax):
                        if 0 + buffval <= j <= 2048 - buffval:
                            ij = np.array([i,j])
                            dist = np.linalg.norm(ij - np.array((x,y)))
                            i, j = int(i), int(j)
                            distnum = 3
                            if dist > rmin and dist <= rmax:
                                xlist.append(i)
                                ylist.append(j)
                                
            mc_spec.append(np.nanmedian(incube[:, xlist, ylist], axis=1))
        
        mc_spec = np.array(mc_spec)
        mc_ave = np.mean(mc_spec, axis=0)
        percent_diff = np.mean((outspec - mc_ave) / outspec) * 100
        
        return outspec, inpix, percent_diff

    return outspec, inpix


### Function to reproject the data cube using MontagePy

def reproject_sit(indata, in_sit_header, process_num, galaxy, header_file):

    sit_rep_out = []
    pn = f'{process_num}'

    for i in range(indata.shape[2]):
        
        ### Make fits file for a wavelength channel
        fits.PrimaryHDU(data = indata[:,:,i], header = in_sit_header).writeto(f"/home/habjan/SITELLE/data/Montage_data/{galaxy}_SITELLE_wave_channel_{i}_{pn}.fits", overwrite=True) 

        ### Reproject and save using MontagePy
        mProject(f"/home/habjan/SITELLE/data/Montage_data/{galaxy}_SITELLE_wave_channel_{i}_{pn}.fits", 
                 f"/home/habjan/SITELLE/data/Montage_data/{galaxy}_SITELLE_wave_channel_mp_{i}_{pn}.fits",  ### mp for muse projection
                 header_file,
                 fullRegion=True, energyMode=False)
        
        ### Import
        oii_flux_mp_i = fits.open(f"/home/habjan/SITELLE/data/Montage_data/{galaxy}_SITELLE_wave_channel_mp_{i}_{pn}.fits")[0].data
    
        ### Open and store reprojected wavelength channel
        sit_rep_out.append(oii_flux_mp_i)

    return sit_rep_out

### [OII]7319 Error Function

def mcerr7319(inwave, inputflux, gnoise, wave0, iters, insig):
    
    def gaussian(x, a, red, sig):    #a:amplitude, x0:average wavelength, sigma:spectral resolution, C:zero offeset
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2)) + C + lp*x
    
    def gaussian_noC(x, a, red, sig):
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2))
    
    fluxerr = np.zeros(iters)
    
    wavelength = 7320
    c = 3 * 10**5
    m = (35-80)/(4750-9350)
    dopv = m*(wavelength - 4750) + 35
    sigma = wavelength * (dopv / c)
    
    for i in range(len(fluxerr)):
        noiseflux = inputflux + np.random.normal(0, gnoise, len(inputflux))
        lowb = wave0 - 10
        highb = wave0 + 5   
        low = np.where(inwave > lowb)[0][0]
        up = np.where(inwave > highb)[0][0]

        noise_ind = np.where(((inwave > wave0 - 150) & (inwave < wave0 - 10)) | ((inwave > wave0 + 20) & (inwave < wave0 + 150)))[0]
        fit_lam = inwave[noise_ind]
        fit_spec = noiseflux[noise_ind]
        fit = np.polyfit(fit_lam, fit_spec, 1)
        lp, C = fit[0], fit[1]

        ind1 = [np.where(inwave > wave0 - 150)[0][0], np.where(inwave > wave0 - 10)[0][0]]
        ind2 = [np.where(inwave > wave0 + 20)[0][0], np.where(inwave > wave0 + 150)[0][0]]

        waves = inwave[low:up]
        fluxes = noiseflux[low:up]
        p0list = np.array([500, wave0, insig])
        
        try:
            param, paramcov = curve_fit(f=gaussian, xdata=waves, ydata=fluxes, p0=p0list, 
                                        bounds=(np.array([0, wave0-0.5, insig]), 
                                                np.array([np.inf, wave0+0.5, insig+0.5])))
            
            flux = quad(gaussian_noC, param[1] - 10 * insig, param[1] + 10 * insig, args=(param[0], param[1], param[2]))[0]
        
            fluxerr[i] = flux
        except:
            fluxerr[i] = np.nan
            
    err = np.array([fluxerr[i] for i in range(len(fluxerr)) if np.isnan(fluxerr[i]) == False])
    err = np.std(err)
    
    return err

### [OII]7319 Refit Function

def refit7319(inwave, influx, snval, phdata, galvel, mcit, mwebv):
    
    def gaussian(x, a, red, sig):    #a:amplitude, x0:average wavelength, sigma:spectral resolution, C:zero offeset
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2)) + C + lp*x
    
    def gaussian_noC(x, a, red, sig):
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2))

    wavelength = 7320
    c = 3 * 10**5
    m = (35-80)/(4750-9350)
    dopv = m*(wavelength - 4750) + 35
    sigma = wavelength * (dopv / c)
    
    outflux = np.zeros(len(influx))
    outfluxerr = np.zeros(len(influx))
    wave_out = np.zeros(len(influx))
    param_list = []
    snr_arr = np.zeros(len(influx))

    for i in range(len(influx)):
        w0 = wavelength*(phdata[i]['NII6583_VEL']+galvel)/(299792) + wavelength
        sigma = wavelength * (np.array(phdata['NII6583_SIGMA'])[i] / c)

        if np.array(phdata['NII6583_SIGMA'])[i] / np.array(phdata['NII6583_SIGMA_ERR'])[i] < snval:
            outflux[i], outfluxerr[i] = np.nan, np.nan
            param_list.append(np.array([np.nan for j in range(5)]))
            snr_arr[i] = np.nan
            continue

        lowb = w0 - 10
        highb = w0 + 5        #plus five to not include [O II]7330 peak
        low = np.where(inwave > lowb)[0][0]
        up = np.where(inwave > highb)[0][0]

        noise_ind = np.where(((inwave > w0 - 150) & (inwave < w0 - 10)) | ((inwave > w0 + 20) & (inwave < w0 + 150)))[0]
        fit_lam = inwave[noise_ind]
        fit_spec = influx[i][noise_ind]
        fit = np.polyfit(fit_lam, fit_spec, 1)
        lp, C = fit[0], fit[1]

        ind1 = [np.where(inwave > w0 - 150)[0][0], np.where(inwave > w0 - 10)[0][0]]
        ind2 = [np.where(inwave > w0 + 20)[0][0], np.where(inwave > w0 + 150)[0][0]]
        noise1 = np.std(influx[i][ind1[0]:ind1[1]])
        noise2 = np.std(influx[i][ind2[0]:ind2[1]])
        noise = np.mean([noise1, noise2])

        waves = inwave[low:up]
        fluxes = influx[i][low:up]
        p0list = np.array([500, w0, sigma])
        wave_out[i] = w0
        
        try:
            param, paramcov = curve_fit(f=gaussian, xdata=waves, ydata=fluxes, p0=p0list, 
                                        bounds=(np.array([0, w0-0.5, sigma]), 
                                                np.array([np.inf, w0+0.5, sigma+0.5])))
            wave_out[i] = param[1]
            param_list.append(np.concatenate([param, [C, lp]]))
        except:
            outflux[i], outfluxerr[i] = np.nan, np.nan
            param_list.append(np.array([np.nan for j in range(len(p0list) + 2)]))
            snr_arr[i] = np.nan
            continue
            
        flux = quad(gaussian_noC, param[1] - 10 * sigma, param[1] + 10 * sigma, args=(param[0], param[1], param[2]))[0]
        signal = np.nanmax(gaussian(inwave[(param[1] - 10 < inwave) & (param[1] + 10 > inwave)], param[0], param[1], param[2]))
        snr_arr[i] = signal/noise
        
        if signal/noise > snval and flux > 0: 
            outflux[i] = flux
            outfluxerr[i] = mcerr7319(inwave, influx[i], noise, w0, mcit, sigma)
        else: 
            outflux[i] = np.nan
            outfluxerr[i] = np.nan
            continue
    
    R_V = 3.1
    mwcorr7319 = remove(extinction.odonnell94(wave_out, mwebv*R_V, R_V), outflux)
    mwcorr7319err = remove(extinction.odonnell94(wave_out, mwebv*R_V, R_V), outfluxerr)

    return mwcorr7319, mwcorr7319err, wave_out, np.array(param_list), snr_arr

### [OII]7330 Error Function

def mcerr7330(inwave, inputflux, gnoise, wave0, iters, insig):
    
    def gaussian(x, a, red, sig):    #a:amplitude, wavelength:feature wavelength, sigma:spectral resolution, C:zero offeset
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2)) + C + lp*x
    
    def gaussian_noC(x, a, red, sig): 
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2))
    
    fluxerr = np.zeros(iters)
    
    wave = 7330
    c = 3 * 10**5
    m = (35-80)/(4750-9350)
    dopv = m*(wave - 4750) + 35
    sigma = wave * (dopv / c)
    
    for i in range(len(fluxerr)):
        noiseflux = inputflux + np.random.normal(0, gnoise, len(inputflux))
        lowb = wave0 - 5
        highb = wave0 + 10  

        noise_ind = np.where(((inwave > wave0 - 150) & (inwave < wave0 - 20)) | ((inwave > wave0 + 10) & (inwave < wave0 + 150)))[0]
        fit_lam = inwave[noise_ind]
        fit_spec = noiseflux[noise_ind]
        fit = np.polyfit(fit_lam, fit_spec, 1)
        lp, C = fit[0], fit[1]

        ind1 = [np.where(inwave > wave0 - 150)[0][0], np.where(inwave > wave0 - 20)[0][0]]
        ind2 = [np.where(inwave > wave0 + 10)[0][0], np.where(inwave > wave0 + 150)[0][0]]

        low = np.where(inwave > lowb)[0][0]
        up = np.where(inwave > highb)[0][0]
        waves = inwave[low:up]
        fluxes = noiseflux[low:up]
        p0list = np.array([500, wave0, insig])
        
        try:
            param, paramcov = curve_fit(f=gaussian, xdata=waves, ydata=fluxes, p0=p0list, 
                                        bounds=(np.array([0, wave0-0.5, insig]), 
                                                np.array([np.inf, wave0+0.5, insig+0.5])))
            
            flux = quad(gaussian_noC, param[1] - 10 * insig, param[1] + 10 * insig, args=(param[0], param[1], param[2]))[0]
        
            fluxerr[i] = flux
        except:
            fluxerr[i] = np.nan
            
    err = np.array([fluxerr[i] for i in range(len(fluxerr)) if np.isnan(fluxerr[i]) == False])
    err = np.std(err)
    
    return err

### [OII]7330 Refit Function

def refit7330(inwave, influx, snval, indata, galvel, mcit, mwebv):
    outflux = np.zeros(len(influx))
    outfluxerr = np.zeros(len(influx))
    wave_out = np.zeros(len(influx))

    param_list = []
    snr_arr = np.zeros(len(influx))

    wave = 7330
    c = 3 * 10**5
    m = (35-80)/(4750-9350)
    dopv = m*(wave - 4750) + 35
    sigma = wave * (dopv / c)
    
    def gaussian(x, a, red, sig):    #a:amplitude, wavelength:feature wavelength, sigma:spectral resolution, C:zero offeset
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2)) + C + lp*x
    
    def gaussian_noC(x, a, red, sig): 
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2))
    
    for i in range(len(influx)):
        w0 = wave*(indata[i]['NII6583_VEL']+galvel)/(299792) + wave
        sigma = wave * (np.array(indata['NII6583_SIGMA'])[i] / c)

        if np.array(indata['NII6583_SIGMA'])[i] / np.array(indata['NII6583_SIGMA_ERR'])[i] < snval:
            outflux[i], outfluxerr[i] = np.nan, np.nan
            param_list.append(np.array([np.nan for j in range(5)]))
            snr_arr[i] = np.nan
            continue

        lowb = w0 - 5
        highb = w0 + 10
    
        noise_ind = np.where(((inwave > w0 - 150) & (inwave < w0 - 20)) | ((inwave > w0 + 10) & (inwave < w0 + 150)))[0]
        fit_lam = inwave[noise_ind]
        fit_spec = influx[i][noise_ind]
        fit = np.polyfit(fit_lam, fit_spec, 1)
        lp, C = fit[0], fit[1]

        ind1 = [np.where(inwave > w0 - 150)[0][0], np.where(inwave > w0 - 20)[0][0]]
        ind2 = [np.where(inwave > w0 + 10)[0][0], np.where(inwave > w0 + 150)[0][0]]
        noise1 = np.std(influx[i][ind1[0]:ind1[1]])
        noise2 = np.std(influx[i][ind2[0]:ind2[1]])
        noise = np.mean([noise1, noise2])

        waves = inwave[np.where(inwave > lowb)[0][0]:np.where(inwave > highb)[0][0]]
        fluxes = influx[i][np.where(inwave > lowb)[0][0]:np.where(inwave > highb)[0][0]]
        p0list = np.array([500, w0, sigma])
        wave_out[i] = w0
        
        try:
            param, paramcov = curve_fit(f=gaussian, xdata=waves, ydata=fluxes, p0=p0list, 
                                        bounds=(np.array([0, w0-0.5, sigma]), 
                                                np.array([np.inf, w0+0.5, sigma+0.5])))
            wave_out[i] = param[1]
            param_list.append(np.concatenate([param, [C, lp]]))
        except:
            outflux[i], outfluxerr[i] = np.nan, np.nan
            param_list.append(np.array([np.nan for j in range(len(p0list) + 2)]))
            snr_arr[i] = np.nan
            continue
            
        flux = quad(gaussian_noC, param[1] - 10 * sigma, param[1] + 10 * sigma, args=(param[0], param[1], param[2]))[0]
        signal = np.nanmax(gaussian(inwave[(param[1] - 10 < inwave) & (param[1] + 10 > inwave)], param[0], param[1], param[2]))
        snr_arr[i] = signal/noise

        if signal/noise > snval and flux > 0:
            outflux[i] = flux
            outfluxerr[i] = mcerr7330(inwave, influx[i], noise, w0, mcit, sigma)
        else: 
            outflux[i] = np.nan
            outfluxerr[i] = np.nan
            continue
    
    R_V = 3.1
    mwcorr7330 = remove(extinction.odonnell94(wave_out, mwebv*R_V, R_V), outflux)
    mwcorr7330err = remove(extinction.odonnell94(wave_out, mwebv*R_V, R_V), outfluxerr)

    return mwcorr7330, mwcorr7330err, wave_out, np.array(param_list), snr_arr

### [NII]5754 Error Function

def mcerr5755(inwave, inputflux, gnoise, wave0, iters, galaxy_in, insig):
    
    def gaussian(x, a, red, sig):    #a:amplitude, x0:average wavelength, sigma:spectral resolution, C:zero offeset
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2)) + C + lp*x
    
    def gaussian_noC(x, a, red, sig):
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2))
    
    fluxerr = np.zeros(iters)

    wavelength = 5755
    c = 3 * 10**5
    m = (35-80)/(4750-9350)
    dopv = m*(wavelength - 4750) + 35
    sigma = wavelength * (dopv / c)
    
    for i in range(len(fluxerr)):
        noiseflux = inputflux + np.random.normal(0, gnoise, len(inputflux))
        lowb = wave0 - 10
        highb = wave0 + 2*insig
        low = np.where(inwave > lowb)[0][0]
        up = np.where(inwave > highb)[0][0]

        if galaxy_in in np.array(['NGC4535', 'NGC4254']):

            noise_ind = np.where(((inwave > wave0 - 175) & (inwave < wave0 - 8)) | ((inwave > wave0 + 180) & (inwave < wave0 + 400)))[0]
            fit_lam = inwave[noise_ind]
            fit_spec = noiseflux[noise_ind]
            fit = np.polyfit(fit_lam, fit_spec, 1)
            lp, C = fit[0], fit[1]

            ind1 = [np.where(inwave > wave0 - 175)[0][0], np.where(inwave > wave0 - 8)[0][0]]
            ind2 = [np.where(inwave > wave0 + 180)[0][0], np.where(inwave > wave0 + 400)[0][0]]

        else: 

            noise_ind = np.where(((inwave > wave0 - 175) & (inwave < wave0 - 8)) | ((inwave > wave0 + 180) & (inwave < wave0 + 400)) | ((inwave > wave0 + 8) & (inwave < wave0 + 100)))[0]
            fit_lam = inwave[noise_ind]
            fit_spec = noiseflux[noise_ind]
            fit = np.polyfit(fit_lam, fit_spec, 1)
            lp, C = fit[0], fit[1]

            ind1 = [np.where(inwave > wave0 - 175)[0][0], np.where(inwave > wave0 - 8)[0][0]]
            ind2 = [np.where(inwave > wave0 + 8)[0][0], np.where(inwave > wave0 + 100)[0][0]]
            ind3 = [np.where(inwave > wave0 + 180)[0][0], np.where(inwave > wave0 + 400)[0][0]]

        waves = inwave[low:up]
        fluxes = noiseflux[low:up]
        p0list = np.array([500, wave0, insig])
        
        try:
            param, paramcov = curve_fit(f=gaussian, xdata=waves, ydata=fluxes, p0=p0list, 
                                        bounds=(np.array([0, wave0-0.5, insig]), 
                                                np.array([np.inf, wave0+0.5, insig+0.5])))
            
            flux = quad(gaussian_noC, param[1] - 10 * insig, param[1] + 10 * insig, args=(param[0], param[1], param[2]))[0]
        
            fluxerr[i] = flux
        except:
            fluxerr[i] = np.nan
            
    err = np.array([fluxerr[i] for i in range(len(fluxerr)) if ~np.isnan(fluxerr[i])])
    err = np.std(err)
    
    return err

### [NII]5754 Refit

def refit5755(inwave, influx,snval, phdata, galvel, mcit, mwebv, ingal):
    
    def gaussian(x, a, red, sig):    #a:amplitude, x0:average wavelength, sigma:spectral resolution, C:zero offeset
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2)) + C + lp*x
    
    def gaussian_noC(x, a, red, sig):
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2))
    
    wavelength = 5755
    c = 3 * 10**5
    m = (35-80)/(4750-9350)
    dopv = m*(wavelength - 4750) + 35
    sigma = wavelength * (dopv / c)

    outflux = np.zeros(len(influx))
    outfluxerr = np.zeros(len(influx))
    wave_out = np.zeros(len(influx))

    param_list = []
    snr_arr = np.zeros(len(influx))

    for i in range(len(influx)):
        w0 = wavelength*(phdata[i]['NII6583_VEL']+galvel)/(299792) + wavelength
        sigma = wavelength * (np.array(phdata['NII6583_SIGMA'])[i] / c)

        if np.array(phdata['NII6583_SIGMA'])[i] / np.array(phdata['NII6583_SIGMA_ERR'])[i] < snval:
            outflux[i], outfluxerr[i] = np.nan, np.nan
            param_list.append(np.array([np.nan for j in range(5)]))
            snr_arr[i] = np.nan
            continue

        lowb = w0 - 10
        highb = w0 + 2*sigma
        low = np.where(inwave > lowb)[0][0]
        up = np.where(inwave > highb)[0][0]

        if ingal in np.array(['NGC4535', 'NGC4254']):

            noise_ind = np.where(((inwave > w0 - 175) & (inwave < w0 - 8)) | ((inwave > w0 + 180) & (inwave < w0 + 400)))[0]
            fit_lam = inwave[noise_ind]
            fit_spec = influx[i][noise_ind]
            fit = np.polyfit(fit_lam, fit_spec, 1)
            lp, C = fit[0], fit[1]

            ind1 = [np.where(inwave > w0 - 175)[0][0], np.where(inwave > w0 - 8)[0][0]]
            ind2 = [np.where(inwave > w0 + 180)[0][0], np.where(inwave > w0 + 400)[0][0]]
            noise1 = np.std(influx[i][ind1[0]:ind1[1]])
            noise2 = np.std(influx[i][ind2[0]:ind2[1]])
            noise = np.mean([noise1, noise2])
            fluxmed = np.median(np.concatenate([influx[i][ind1[0]:ind1[1]], influx[i][ind2[0]:ind2[1]]]))

        else: 

            noise_ind = np.where(((inwave > w0 - 175) & (inwave < w0 - 8)) | ((inwave > w0 + 180) & (inwave < w0 + 400)) | ((inwave > w0 + 8) & (inwave < w0 + 100)))[0]
            fit_lam = inwave[noise_ind]
            fit_spec = influx[i][noise_ind]
            fit = np.polyfit(fit_lam, fit_spec, 1)
            lp, C = fit[0], fit[1]
            
            ind1 = [np.where(inwave > w0 - 175)[0][0], np.where(inwave > w0 - 8)[0][0]]
            ind2 = [np.where(inwave > w0 + 8)[0][0], np.where(inwave > w0 + 100)[0][0]]
            ind3 = [np.where(inwave > w0 + 180)[0][0], np.where(inwave > w0 + 400)[0][0]]
            noise1 = np.std(influx[i][ind1[0]:ind1[1]])
            noise2 = np.std(influx[i][ind2[0]:ind2[1]])
            noise = np.mean([noise1, noise2])
            fluxmed = np.median(np.concatenate([influx[i][ind1[0]:ind1[1]], influx[i][ind2[0]:ind2[1]]]))

        waves = inwave[low:up]
        fluxes = influx[i][low:up]
        p0list = np.array([500, w0, sigma])
        wave_out[i] = w0
        
        try:
            param, paramcov = curve_fit(f=gaussian, xdata=waves, ydata=fluxes, p0=p0list, 
                                        bounds=(np.array([0, w0-0.5, sigma]), 
                                                np.array([np.inf, w0+0.5, sigma+0.5])))
            wave_out[i] = param[1]
            param_list.append(np.concatenate([param, [C, lp]]))
        except:
            outflux[i], outfluxerr[i] = np.nan, np.nan
            param_list.append(np.array([np.nan for j in range(len(p0list) + 2)]))
            snr_arr[i] = np.nan
            continue
            
        flux = quad(gaussian_noC, param[1] - 10 * sigma, param[1] + 10 * sigma, args=(param[0], param[1], param[2]))[0]
        signal = np.nanmax(gaussian(inwave[(param[1] - 10 < inwave) & (param[1] + 10 > inwave)], param[0], param[1], param[2]))
        snr_arr[i] = signal/noise
        
        if signal/noise > snval and flux > 0:  
            outflux[i] = flux
            outfluxerr[i] = mcerr5755(inwave, influx[i], noise, w0, mcit, ingal, sigma)
        else: 
            outflux[i] = np.nan
            outfluxerr[i] = np.nan
            continue
    
    R_V = 3.1
    mwcorr5755 = remove(extinction.odonnell94(wave_out, mwebv*R_V, R_V), outflux)
    mwcorr5755err = remove(extinction.odonnell94(wave_out, mwebv*R_V, R_V), outfluxerr)
    
    return mwcorr5755, mwcorr5755err, wave_out, np.array(param_list), snr_arr

### [SIII]6312 Error Function

def mcerr6312(inwave, inputflux, gnoise, wave0, iters, insig):
    
    def gaussian(x, a, x0, sig):    #a:amplitude, wave0:feature wavelength, sigma:spectral resolution, C:zero offeset
        return a * np.exp((-(x-x0) ** 2)/ (2 * sig ** 2)) + C + lp*x
    
    def gaussian_noC(x, a, x0, sig): 
        return a * np.exp((-(x-x0) ** 2)/ (2 * sig ** 2))
    
    fluxerr = np.zeros(iters)
    
    wave = 6312
    c = 3 * 10**5
    m = (35-80)/(4750-9350)
    dopv = m*(wave - 4750) + 35
    sigma = wave * (dopv / c)
    
    for i in range(len(fluxerr)):
        noiseflux = inputflux + np.random.normal(0, gnoise, len(inputflux))
        lowb = wave0 - 7*insig
        highb = wave0 + 12  
        low = np.where(inwave > lowb)[0][0]
        up = np.where(inwave > highb)[0][0]

        noise_ind = np.where(((inwave > wave0 - 200) & (inwave < wave0 - 70)) | ((inwave > wave0 + 63) & (inwave < wave0 + 120)) | ((inwave > wave0 + 17) & (inwave < wave0 + 45)))[0]
        fit_lam = inwave[noise_ind]
        fit_spec = noiseflux[noise_ind]
        fit = np.polyfit(fit_lam, fit_spec, 1)
        lp, C = fit[0], fit[1]

        ind1 = [np.where(inwave > wave0 - 200)[0][0], np.where(inwave > wave0 - 70)[0][0]]
        ind2 = [np.where(inwave > wave0 + 63)[0][0], np.where(inwave > wave0 + 120)[0][0]]
        ind3 = [np.where(inwave > wave0 + 17)[0][0], np.where(inwave > wave0 + 45)[0][0]]

        waves = inwave[low:up]
        fluxes = noiseflux[low:up]
        p0list = np.array([500, wave0, insig])
        
        try:
            param, paramcov = curve_fit(f=gaussian, xdata=waves, ydata=fluxes, p0=p0list, 
                                        bounds=(np.array([0, wave0-0.5, insig]), 
                                                np.array([np.inf, wave0+0.5, insig+0.5])))
            
            flux = quad(gaussian_noC, param[1] - 10 * insig, param[1] + 10 * insig, args=(param[0], param[1], param[2]))[0]
        
            fluxerr[i] = flux
        except:
            fluxerr[i] = np.nan
            
    err = np.array([fluxerr[i] for i in range(len(fluxerr)) if np.isnan(fluxerr[i]) == False])
    err = np.std(err)
    
    return err

### [SIII]6312 Refit

def refit6312(inwave, influx, snval, phdata, galvel, mcit, mwebv):
    
    def gaussian(x, a, red, sig):    #a:amplitude, wavelength:feature wavelength, d7330:spectral resolution, C:zero offeset
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2)) + C + lp*x
    
    def gaussian_noC(x, a, red, sig): 
        return a * np.exp((-(x-red) ** 2)/ (2 * sig ** 2))
    
    wavelength = 6312
    c = 3 * 10**5
    m = (35-80)/(4750-9350)
    dopv = m*(wavelength - 4750) + 35
    sigma = wavelength * (dopv / c)

    outflux = np.zeros(len(influx))
    outfluxerr = np.zeros(len(influx))
    wave_out = np.zeros(len(influx))

    param_list = []
    snr_arr = np.zeros(len(influx))

    for i in range(len(influx)):
        w0 = wavelength*(phdata[i]['OIII5006_VEL']+galvel)/(299792) + wavelength
        sigma = wavelength * (np.array(phdata['OIII5006_SIGMA'])[i] / c)

        if np.array(phdata['OIII5006_SIGMA'])[i] / np.array(phdata['OIII5006_SIGMA_ERR'])[i] < snval:
            outflux[i], outfluxerr[i] = np.nan, np.nan
            param_list.append(np.array([np.nan for j in range(5)]))
            snr_arr[i] = np.nan
            continue

        lowb = w0 - 7*sigma
        highb = w0 + 12
        low = np.where(inwave > lowb)[0][0]
        up = np.where(inwave > highb)[0][0]

        noise_ind = np.where(((inwave > w0 - 200) & (inwave < w0 - 70)) | ((inwave > w0 + 63) & (inwave < w0 + 120)) | ((inwave > w0 + 17) & (inwave < w0 + 45)))[0]
        fit_lam = inwave[noise_ind]
        fit_spec = influx[i][noise_ind]
        fit = np.polyfit(fit_lam, fit_spec, 1)
        lp, C = fit[0], fit[1]

        ind1 = [np.where(inwave > w0 - 200)[0][0], np.where(inwave > w0 - 70)[0][0]]
        ind2 = [np.where(inwave > w0 + 63)[0][0], np.where(inwave > w0 + 120)[0][0]]
        ind3 = [np.where(inwave > w0 + 17)[0][0], np.where(inwave > w0 + 45)[0][0]]
        noise1 = np.std(influx[i][ind1[0]:ind1[1]])
        noise2 = np.std(influx[i][ind2[0]:ind2[1]])
        noise3 = np.std(influx[i][ind3[0]:ind3[1]])
        noise = np.mean([noise1, noise2, noise3])

        waves = inwave[low:up]
        fluxes = influx[i][low:up]
        p0list = np.array([500, w0, sigma])
        wave_out[i] = w0
        
        try:
            param, paramcov = curve_fit(f=gaussian, xdata=waves, ydata=fluxes, p0=p0list, 
                                        bounds=(np.array([0, w0-0.5, sigma]), 
                                                np.array([np.inf, w0+0.5, sigma+0.5])))
            wave_out[i] = param[1]
            param_list.append(np.concatenate([param, [C, lp]]))
        except:
            outflux[i], outfluxerr[i] = np.nan, np.nan
            param_list.append(np.array([np.nan for j in range(len(p0list) + 2)]))
            snr_arr[i] = np.nan
            continue

        flux = quad(gaussian_noC, param[1] - 10 * sigma, param[1] + 10 * sigma, args=(param[0], param[1], param[2]))[0]
        signal = np.nanmax(gaussian(inwave[(param[1] - 10 < inwave) & (param[1] + 10 > inwave)], param[0], param[1], param[2]))
        snr_arr[i] = signal/noise
            
        if signal/noise > snval and flux > 0: 
            outflux[i] = flux
            outfluxerr[i] = mcerr6312(inwave, influx[i], noise, w0, mcit, sigma)
        else: 
            outflux[i] = np.nan
            outfluxerr[i] = np.nan
            continue
    
    R_V = 3.1
    mwcorr6312 = remove(extinction.odonnell94(wave_out, mwebv*R_V, R_V), outflux)
    mwcorr6312err = remove(extinction.odonnell94(wave_out, mwebv*R_V, R_V), outfluxerr)
    
    return mwcorr6312, mwcorr6312err, wave_out, np.array(param_list), snr_arr

### [OII]3727 MC error function

def mcerr3727(inputflux, inwave, sitcube, gnoise, redwave_in, iters, galactic_vel):
    
    flux3727err = np.zeros(iters)
    velocity3727err = np.zeros(iters)

    fitshape = 'sincgauss'
    ang3726, ang3729 = Lines().get_line_nm('[OII]3726') * 10, Lines().get_line_nm('[OII]3729') * 10
    wave1 = ((ang3729 - ang3726) / 1.4) + ang3726 
    #OII3726 = Lines().get_line_cm1('[OII]3726')
    OII3729 = Lines().get_line_cm1('[OII]3729')
    
    fwhm0 = sitcube.get_header()['HIERARCH line_fwhm']
    sigma0 = fwhm0 / (2 * np.sqrt(2 * np.log(2) ))
    
    for i in range(iters):
        spec = inputflux + np.random.normal(0, gnoise, len(inputflux))

        n1 = [np.where(inwave > redwave_in - 450)[0][0],np.where(inwave > redwave_in - 175)[0][0]]
        n2 = [np.where(inwave > redwave_in + 175)[0][0], np.where(inwave > redwave_in + 700)[0][0]]

        fit_lam = np.concatenate([inwave[n2[0]:n2[1]], inwave[n1[0]:n1[1]]])
        fit_spec = np.concatenate([spec[n2[0]:n2[1]], spec[n1[0]:n1[1]]])
        s_fit = np.polyfit(fit_lam, fit_spec, 1)
        sfit_func = np.poly1d(s_fit)

        spec = spec - sfit_func(inwave)
        spec[(1/(inwave * 10**-8) < 3650) | (1/(inwave * 10**-8) > 3850)] = 0
        
        try:
            fit = fit_lines_in_spectrum(spectrum = spec, 
                                        lines = [redwave_in], 
                                        step=sitcube.params.step, 
                                        order=sitcube.params.order, 
                                        nm_laser=sitcube.params.nm_laser, 
                                        theta=corr2theta(sitcube.params.axis_corr), 
                                        zpd_index=sitcube.params.zpd_index, 
                                        wavenumber=True, 
                                        apodization=1, 
                                        fmodel=fitshape, 
                                        sigma_def='free',
                                        sigma_guess = [sigma0])
            
            wave2 = 1/(fit['lines_params'][0][2] * 10**-8)
            
            flux3727err[i], velocity3727err[i] = fit['flux'], ((((wave2-wave1)/wave1)*(3*10**8)) * 10**-3) - galactic_vel
        except:
            flux3727err[i], velocity3727err[i] = np.nan, np.nan

    err3727 = np.array([flux3727err[i] for i in range(len(flux3727err)) if ~np.isnan(flux3727err[i])])
    err3727 = np.std(err3727)

    velerr3727 = np.array([velocity3727err[i] for i in range(len(velocity3727err)) if ~np.isnan(velocity3727err[i])])
    velerr3727 = np.std(velerr3727)
    
    return err3727, velerr3727

### [OII]3727 Flux fit (using ORCS)

def fit_3727(phdata, inspec, inwave, incube, ingalvel, inebv, insnval, mcit):

    out_fluxes = np.zeros(len(phdata))
    out_fluxes_err = np.zeros(len(phdata))
    out_velocity = np.zeros(len(phdata))
    out_velocity_err = np.zeros(len(phdata))
    out_wave = np.zeros(len(phdata))

    param_list = []
    snr_arr = np.zeros(len(phdata))

    fitshape = 'sincgauss'
    OII3726 = Lines().get_line_cm1('[OII]3726')
    OII3729 = Lines().get_line_cm1('[OII]3729')
    ang3726, ang3729 = Lines().get_line_nm('[OII]3726') * 10, Lines().get_line_nm('[OII]3729') * 10
    wave1 = ((ang3729 - ang3726) / 1.4) + ang3726 
    
    conv = 10**8 / inwave**2
    c = 299792
    fwhm0 = incube.get_header()['HIERARCH line_fwhm']
    sigma0 = fwhm0 / (2 * np.sqrt(2 * np.log(2) ))
    
    for i in range(len(phdata)):
        
        ### Cover flux per angstrom to flux per cm_1
        inspectrum = inspec[i]
        spec_cm = inspectrum #* conv
        
        ### Zero out any nan values
        spec_cm[np.isnan(spec_cm)] = 0
        
        ### Find the reddened center wavelength
        nii_vel = phdata[i]['NII6583_VEL']
        region_vel = np.float64(nii_vel+ingalvel)
        red3729 = wave1 * (1 + (region_vel/c))
        red3729_cm = 1/(red3729 * 10**-8)
        out_wave[i] = 1/(red3729_cm * 10**-8)
        
        ### Find noise regions inside of spectrum
        n1_lower, n1_upper = np.where(inwave > red3729_cm - 450)[0][0], np.where(inwave > red3729_cm - 175)[0][0]
        n2_lower, n2_upper = np.where(inwave > red3729_cm + 175)[0][0], np.where(inwave > red3729_cm + 700)[0][0]

        ### Fit a line to the noise
        fit_lam = np.concatenate([inwave[n2_lower: n2_upper], inwave[n1_lower: n1_upper]])
        fit_spec = np.concatenate([spec_cm[n2_lower: n2_upper], spec_cm[n1_lower: n1_upper]])
        s_fit = np.polyfit(fit_lam, fit_spec, 1)
        sfit_func = np.poly1d(s_fit)
        
        # Calculate the standard deviation of the noise and subtract linear fit from spectrum
        noisestd = (np.std(spec_cm[n2_lower: n2_upper]) + np.std(spec_cm[n1_lower: n1_upper])) / 2
        sub_spec_cm = spec_cm - sfit_func(inwave)
        
        # Set any values outside of the filter range equal to zero
        sn1_filter_lower_bound = 1/(inwave * 10**-8) < 3650
        sn1_filter_upper_bound = 1/(inwave * 10**-8) > 3850
        sub_spec_cm[(sn1_filter_lower_bound) | (sn1_filter_upper_bound)] = 0
        
        try:
            fit = fit_lines_in_spectrum(spectrum = sub_spec_cm, 
                                        lines = [red3729_cm],
                                        step = incube.params.step, 
                                        order = incube.params.order, 
                                        nm_laser = incube.params.nm_laser,
                                        theta = corr2theta(incube.params.axis_corr), 
                                        zpd_index = incube.params.zpd_index, 
                                        wavenumber = True,
                                        apodization = 1, 
                                        fmodel = fitshape,
                                        sigma_def='free',
                                        sigma_guess = [sigma0])
            
            snr = fit['lines_params'][0][1] / noisestd

            param_list.append(fit['fitted_models']['Cm1LinesModel'][0])
            snr_arr[i] = snr

            if snr > insnval:
                flux = float(fit['flux'][0])
                redline = 1/(float(fit['lines_params'][0][2]) * 10**-8)

                wave2 = 1/(fit['lines_params'][0][2] * 10**-8)

                out_fluxes[i], out_velocity[i] = flux, ((((wave2-wave1)/wave1)*(3*10**8)) * 10**-3) - ingalvel

                out_fluxes_err[i], out_velocity_err[i] = mcerr3727(sub_spec_cm, inwave, incube, noisestd, red3729_cm, mcit, ingalvel)
            else:
                out_fluxes[i] = np.nan
                out_fluxes_err[i] = np.nan
                out_velocity[i] = np.nan
                out_velocity_err[i] = np.nan
        except Exception as e:
            print(f"An error occurred: {e}")
            out_fluxes[i] = np.nan
            out_fluxes_err[i] = np.nan
            out_velocity[i] = np.nan
            out_velocity_err[i] = np.nan
            param_list.append(np.array([np.nan for j in range(len(inwave))]))
            snr_arr[i] = np.nan
            continue
    
    R_V = 3.1
    mwcorr = remove(extinction.odonnell94(out_wave, inebv*R_V, R_V), out_fluxes)
    mwcorr_err = remove(extinction.odonnell94(out_wave, inebv*R_V, R_V), out_fluxes_err)

    return mwcorr * 10**20, mwcorr_err * 10**20, out_velocity, out_velocity_err, out_wave, np.array(param_list), snr_arr

### Correct for extinction

def corr(indata, in_w_7319, in_w_7330, in_w_5755, in_w_6312, in_w_3727):
    
    R_V = 3.1 
    A_V = np.array([indata[i]['EBV'] for i in range(len(indata))]) * R_V

    corr7319 = np.array([remove(extinction.odonnell94(np.array([in_w_7319[i]]), A_V[i], R_V), indata[i]['OII7319_FLUX_REFIT'])[0] for i in range(len(indata))])
    corr7319err = np.array([remove(extinction.odonnell94(np.array([in_w_7319[i]]), A_V[i], R_V), indata[i]['OII7319_FLUX_REFIT_ERR'])[0] for i in range(len(indata))])

    corr7330 = np.array([remove(extinction.odonnell94(np.array([in_w_7330[i]]), A_V[i], R_V), indata[i]['OII7330_FLUX_REFIT'])[0] for i in range(len(indata))])
    corr7330err = np.array([remove(extinction.odonnell94(np.array([in_w_7330[i]]), A_V[i], R_V), indata[i]['OII7330_FLUX_REFIT_ERR'])[0] for i in range(len(indata))])

    corr5755 = np.array([remove(extinction.odonnell94(np.array([in_w_5755[i]]), A_V[i], R_V), indata[i]['NII5754_FLUX_REFIT'])[0] for i in range(len(indata))])
    corr5755err = np.array([remove(extinction.odonnell94(np.array([in_w_5755[i]]), A_V[i], R_V), indata[i]['NII5754_FLUX_REFIT_ERR'])[0] for i in range(len(indata))])
    
    corr6312 = np.array([remove(extinction.odonnell94(np.array([in_w_6312[i]]), A_V[i], R_V), indata[i]['SIII6312_FLUX_REFIT'])[0] for i in range(len(indata))])
    corr6312err = np.array([remove(extinction.odonnell94(np.array([in_w_6312[i]]), A_V[i], R_V), indata[i]['SIII6312_FLUX_REFIT_ERR'])[0] for i in range(len(indata))])

    corr3727 = np.array([remove(extinction.odonnell94(np.array([in_w_3727[i]]), A_V[i], R_V), indata[i]['OII3727_FLUX'])[0] for i in range(len(indata))])
    corr3727err = np.array([remove(extinction.odonnell94(np.array([in_w_3727[i]]), A_V[i], R_V), indata[i]['OII3727_FLUX_ERR'])[0] for i in range(len(indata))])
    
    indata.add_columns([corr5755, corr6312, corr7319, corr7330, corr5755err, corr6312err, corr7319err, corr7330err, corr3727, corr3727err], 
                       names=('NII5754_FLUX_CORR_REFIT', 'SIII6312_FLUX_CORR_REFIT', 'OII7319_FLUX_CORR_REFIT', 'OII7330_FLUX_CORR_REFIT',
                             'NII5754_FLUX_CORR_REFIT_ERR', 'SIII6312_FLUX_CORR_REFIT_ERR', 'OII7319_FLUX_CORR_REFIT_ERR', 'OII7330_FLUX_CORR_REFIT_ERR',
                             'OII3727_FLUX_CORR', 'OII3727_FLUX_CORR_ERR'))

    return indata

### General Density and Temeperature function 

def den_temp_gen(mc_len, bpt_nii, bpt_oi, bpt_sii, temp_eq, den_eq, temp_ratio, den_ratio, temp_ratio_err, den_ratio_err):

    diags = pn.Diagnostics()

    diags.addDiag('[OII] b3727/b7325', ('O2', '(L(3726)+L(3729))/(L(7319)+L(7320)+L(7330)+L(7331))', 
                                        'RMS([E(3727A+),E(7319A+)*L(7319A+)/(L(7319A+)+L(7330A+)),E(7330A+)*L(7330A+)/(L(7319A+)+L(7330A+))])'))

    if bpt_nii != 0 or bpt_oi != 0 or bpt_sii != 0 or np.isnan(temp_ratio) or np.isnan(den_ratio):
        return np.nan, np.nan, np.nan, np.nan
    
    temp, den = diags.getCrossTemDen(diag_tem = temp_eq, diag_den = den_eq, 
                                     value_tem = temp_ratio, 
                                     value_den = den_ratio, 
                                     guess_tem = 10**4)
    if np.isnan(temp) or np.isnan(den):
        return np.nan, np.nan, np.nan, np.nan
    
    temp_mc_arr, den_mc_arr = [], []
    
    for i in range(mc_len):
        t_mc, d_mc = diags.getCrossTemDen(diag_tem = temp_eq, diag_den = den_eq, 
                                          value_tem = np.random.normal(temp_ratio, temp_ratio_err), 
                                          value_den = np.random.normal(den_ratio, den_ratio_err), 
                                          guess_tem = 10**4)
        
        temp_mc_arr.append(t_mc), den_mc_arr.append(d_mc)
    
    temp_err, den_err = np.nanstd(temp_mc_arr), np.nanstd(den_mc_arr)

    return temp, den, temp_err, den_err

def den_oii_func(mc_len, ratio, ratio_err, temp, temp_err):
    
    O2 = pn.Atom('O', 2)

    if np.isnan(temp) or np.isnan(ratio):
        return np.nan, np.nan

    oii_den = O2.getTemDen(int_ratio = ratio, tem=temp, to_eval = '(L(3729) + L(3726)) / (L(7319) + L(7320) + L(7330) + L(7331))')

    if np.isnan(oii_den):
        return np.nan, np.nan
    
    mc_oii_den = np.array([O2.getTemDen(int_ratio = np.random.normal(ratio, ratio_err), tem=np.random.normal(temp, temp_err), to_eval = '(L(3729) + L(3726)) / (L(7319) + L(7320) + L(7330) + L(7331))') for mc in range(mc_len)])

    return oii_den, np.nanstd(mc_oii_den)

### Temperature and Denisty Function

def dentemp(indata, err, iters):

    ### Compile necessary data in arrays
    bpt_nii_arr = np.array(indata['BPT_NII'])
    bpt_sii_arr = np.array(indata['BPT_SII'])
    bpt_oi_arr = np.array(indata['BPT_OI'])

    oii3727 = np.array(indata['OII3727_FLUX_CORR'])
    oii3727_err = np.array(indata['OII3727_FLUX_CORR_ERR'])

    sii6730 = np.array(indata['SII6730_FLUX_CORR'])
    sii6730_err = np.array(indata['SII6730_FLUX_CORR_ERR'])
    sii6716 = np.array(indata['SII6716_FLUX_CORR'])
    sii6716_err = np.array(indata['SII6716_FLUX_CORR_ERR'])

    oii7320 = np.array(indata['OII7319_FLUX_CORR_REFIT'])
    oii7320_err = np.array(indata['OII7319_FLUX_CORR_REFIT_ERR'])
    oii7330 = np.array(indata['OII7330_FLUX_CORR_REFIT'])
    oii7330_err = np.array(indata['OII7330_FLUX_CORR_REFIT_ERR'])

    siii6312 = np.array(indata['SIII6312_FLUX_CORR_REFIT'])
    siii6312_err = np.array(indata['SIII6312_FLUX_CORR_REFIT_ERR'])
    siii9069 = np.array(indata['SIII9068_FLUX_CORR'])
    siii9069_err = np.array(indata['SIII9068_FLUX_CORR_ERR'])

    nii5755 = np.array(indata['NII5754_FLUX_CORR_REFIT'])
    nii5755_err = np.array(indata['NII5754_FLUX_CORR_REFIT_ERR'])
    nii6583 = np.array(indata['NII6583_FLUX_CORR'])
    nii6583_err = np.array(indata['NII6583_FLUX_CORR_ERR'])


    ### T([OII]) and n([SII]) derivation
    # Error propagation for ratios
    z_total = oii3727/(oii7330+oii7320)
    dom_err = np.sqrt(oii7330_err**2 + oii7320_err**2)
    t_ratio_err = z_total * np.sqrt((dom_err / (oii7330 + oii7320))**2 + (oii3727_err / oii3727)**2)

    z_total = sii6730/sii6716
    n_ratio_err = z_total * np.sqrt((sii6716_err / (sii6716))**2 + (sii6730_err / sii6730)**2)
    # temp, den and uncertainty calculation 
    teoii = np.zeros(len(indata))
    nesii_oii = np.zeros(len(indata))
    teoii_err = np.zeros(len(indata))
    nesii_oii_err = np.zeros(len(indata))
    for i in range(len(indata)): 
        teoii[i], nesii_oii[i], teoii_err[i], nesii_oii_err[i] = den_temp_gen(mc_len = iters, 
                                                                  bpt_nii = bpt_nii_arr[i], bpt_oi = bpt_oi_arr[i], bpt_sii=bpt_sii_arr[i], 
                                                                  temp_eq='[OII] b3727/b7325', den_eq = '[SII] 6731/6716', 
                                                                  temp_ratio = oii3727[i]/(oii7330[i]+oii7320[i]), den_ratio=sii6730[i]/sii6716[i], 
                                                                  temp_ratio_err = t_ratio_err[i], den_ratio_err = n_ratio_err[i])
    


    ### T([SIII]) and n([SII]) derivation
    # Error propagation for ratios
    z_total = siii6312 / siii9069
    t_ratio_err = z_total * np.sqrt((siii6312_err/ (siii6312))**2 + (siii9069_err / siii9069)**2)

    z_total = sii6730/sii6716
    n_ratio_err = z_total * np.sqrt((sii6716_err / (sii6716))**2 + (sii6730_err / sii6730)**2)
    # temp, den and uncertainty calculation 
    tesiii = np.zeros(len(indata))
    nesii_siii = np.zeros(len(indata))
    tesiii_err = np.zeros(len(indata))
    nesii_siii_err = np.zeros(len(indata))
    for i in range(len(indata)): 
        tesiii[i], nesii_siii[i], tesiii_err[i], nesii_siii_err[i] = den_temp_gen(mc_len = iters, 
                                                                  bpt_nii = bpt_nii_arr[i], bpt_oi = bpt_oi_arr[i], bpt_sii=bpt_sii_arr[i], 
                                                                  temp_eq='[SIII] 6312/9069', den_eq = '[SII] 6731/6716', 
                                                                  temp_ratio = siii6312[i] / siii9069[i], den_ratio=sii6730[i]/sii6716[i], 
                                                                  temp_ratio_err = t_ratio_err[i], den_ratio_err = n_ratio_err[i])
        
    ### T([NII]) and n([SII]) derivation
    # Error propagation for ratios
    z_total = nii5755 / nii6583
    t_ratio_err = z_total * np.sqrt((nii5755_err/ (nii5755))**2 + (nii6583_err / nii6583)**2)

    z_total = sii6730/sii6716
    n_ratio_err = z_total * np.sqrt((sii6716_err / (sii6716))**2 + (sii6730_err / sii6730)**2)
    # temp, den and uncertainty calculation 
    tenii = np.zeros(len(indata))
    nesii_nii = np.zeros(len(indata))
    tenii_err = np.zeros(len(indata))
    nesii_nii_err = np.zeros(len(indata))
    for i in range(len(indata)): 
        tenii[i], nesii_nii[i], tenii_err[i], nesii_nii_err[i] = den_temp_gen(mc_len = iters, 
                                                                              bpt_nii = bpt_nii_arr[i], bpt_oi = bpt_oi_arr[i], bpt_sii=bpt_sii_arr[i], 
                                                                              temp_eq='[NII] 5755/6584', den_eq = '[SII] 6731/6716', 
                                                                              temp_ratio = nii5755[i] / nii6583[i], den_ratio=sii6730[i]/sii6716[i], 
                                                                              temp_ratio_err = t_ratio_err[i], den_ratio_err = n_ratio_err[i])
    

    ### n([OII]) derivation
    # Error propagation for ratios
    z_total = oii3727/(oii7330+oii7320)
    dom_err = np.sqrt(oii7330_err**2 + oii7320_err**2)
    n_ratio_err = z_total * np.sqrt((dom_err / (oii7330 + oii7320))**2 + (oii3727_err / oii3727)**2)

    neoii = np.zeros(len(indata))
    neoii_err = np.zeros(len(indata))

    for i in range(len(indata)):
        neoii[i], neoii_err[i] = den_oii_func(mc_len = iters, 
                                              ratio = z_total[i], ratio_err = n_ratio_err[i], 
                                              temp = tenii[i], temp_err = tenii_err[i])

        
    ### Combine different n([SII])    

    nesii = np.zeros(len(indata))
    nesii_err = np.zeros(len(indata))

    for i in range(len(indata)):
        
        if np.isnan(nesii_nii[i]) and np.isnan(nesii_oii[i]) and np.isnan(nesii_siii[i]):
            nesii[i], nesii_err[i] = np.nan, np.nan
            continue
        
        nesii[i] = np.nansum([nesii_nii[i]*(1/nesii_nii_err[i]**2), nesii_oii[i]*(1/nesii_oii_err[i]**2), nesii_siii[i]*(1/nesii_siii_err[i]**2)]) / np.nansum([(1/nesii_nii_err[i]**2), (1/nesii_oii_err[i]**2), (1/nesii_siii_err[i]**2)])
        nesii_err[i] = np.sqrt(np.nansum(np.array([nesii_nii_err[i]**2*(1/nesii_nii_err[i]**2)**2, nesii_oii_err[i]**2*(1/nesii_oii_err[i]**2)**2, nesii_siii_err[i]**2*(1/nesii_siii_err[i]**2)**2])) / np.nansum(np.array([1/nesii_nii_err[i]**2, 1/nesii_oii_err[i]**2, 1/nesii_siii_err[i]**2]))**2 )


    ### T([OIII])

    teoiiib12 = np.zeros(len(indata))
    teoiiib12_err = np.zeros(len(indata))

    for i in range(len(indata)):
        
        if tesiii[i] < 14000:
            teoiiib12[i] = 0.7092 * tesiii[i] + 3609.9
        else: 
            teoiiib12[i], teoiiib12_err[i] = np.nan, np.nan
            continue

        teoiiib12_err[i] = np.nanstd(np.array([0.7092 * np.random.normal(tesiii[i], tesiii_err[i]) + 3609.9 for j in range(iters)]))

    ### T0

    t0 = np.zeros(len(indata))
    t0_err = np.zeros(len(indata))

    for i in range(len(indata)):
        
        if ~np.isnan(tenii[i]):
            t0[i] = 1.17*tenii[i] - 3340
        else: 
            t0[i], t0_err[i] = np.nan, np.nan
            continue

        t0_err[i] = np.nanstd(np.array([1.17 * np.random.normal(tenii[i], tenii_err[i]) - 3340 for j in range(iters)]))

    ### t^2

    t2 = np.zeros(len(indata))
    t2err = np.zeros(len(indata))
    for i in range(len(indata)):
        if ~np.isnan(teoiiib12[i]) and ~np.isnan(tenii[i]):
            delt = teoiiib12[i] - tenii[i]
            t2[i] = 2.9 * 10**-5 * delt + 4.68 * 10**-2
            ### monte carloing of t^2
            t2err[i] = np.nanstd(np.array([2.9 * 10**-5 * (np.random.normal(teoiiib12[i], teoiiib12_err[i]) - np.random.normal(tenii[i], tenii_err[i])) + 4.68 * 10**-2 for j in range(iters)]))
        else: 
            t2[i], t2err[i] = np.nan, np.nan
    
    
    indata.add_columns([teoii, tenii, tesiii, teoiiib12, t0, neoii, nesii, t2,
                        teoii_err, tenii_err, tesiii_err, teoiiib12_err, t0_err, neoii_err, nesii_err, t2err], 
                        names=('OII_TEMP', 'NII_TEMP', 'SIII_TEMP','OIII_TEMP_B12', 'OIII_TEMP_MD23', 
                               'OII_DEN', 'SII_DEN', 't^2_MD23', 
                               'OII_TEMP_ERR', 'NII_TEMP_ERR', 'SIII_TEMP_ERR','OIII_TEMP_B12_ERR', 'OIII_TEMP_MD23_ERR', 
                               'OII_DEN_ERR', 'SII_DEN_ERR', 't^2_MD23_ERR'))
    return indata

### General function for ion abundance calculations and MC uncertainty

def ion_function(pyneb_ion, in_flux, in_temp, in_den, in_eval, in_h, mc_iterations, in_flux_err, in_temp_err, in_den_err, in_h_err):

    if in_den < 100:
        in_den, in_den_err = 100, 100


    ion_abun = pyneb_ion.getIonAbundance(int_ratio = in_flux,
                                         tem = in_temp, 
                                         den= in_den, 
                                         to_eval= in_eval, 
                                         Hbeta = in_h)
    
    if np.isnan(ion_abun):
        return np.nan, np.nan
    
    ion_abun_mc_list = []

    for i in range(mc_iterations):
        
        ion_abun_mc = pyneb_ion.getIonAbundance(int_ratio = np.random.normal(in_flux, in_flux_err),
                                                tem = np.random.normal(in_temp, in_temp_err), 
                                                den= np.random.normal(in_den, in_den_err), 
                                                to_eval = in_eval, 
                                                Hbeta = np.random.normal(in_h, in_h_err))
        
        ion_abun_mc_list.append(ion_abun_mc)
    
    ion_sigma = np.nanstd(ion_abun_mc_list)


    return ion_abun, ion_sigma

### General function for elemental O

def O_elem_function(O2_ion, O3_ion, O2_ion_err, O3_ion_err):
    
    icf = pn.ICF()
    icf_list = ['Ial06_16']
    
    abunlist = {'O2': O2_ion, 'O3': O3_ion}
    abundance = icf.getElemAbundance(abunlist, icf_list)
    Z = 12 + np.log10(abundance['Ial06_16'])

    if np.isnan(Z):
        return np.nan, np.nan
    
    ion_err = np.sqrt(O2_ion_err**2 + O3_ion_err**2)
    Z_err = (1/np.log(10))*(ion_err/abundance['Ial06_16'])

    return Z, Z_err

### N ICF function with MC uncertainty

def ICF_polynomial(w):
    return 0.013 - 0.793*w + 8.177*w**2 - 23.194*w**3 + 26.364*w**4 - 10.536*w**5

def ICF_N(O2_in, O3_in, O2_in_err, O3_in_err, iterations):

    N_ICF = 10 ** ICF_polynomial(O3_in / (O2_in + O3_in))

    if np.isnan(N_ICF):
        return np.nan, np.nan
    
    icf_mc_list = []

    for i in range(iterations):
        try:
            mc_val = 10**ICF_polynomial(np.random.normal(O3_in, O3_in_err) / (np.random.normal(O2_in, O2_in_err) + np.random.normal(O3_in, O3_in_err)))
            icf_mc_list.append(mc_val)
        except:
            continue

    N_ICF_err = np.nanstd(icf_mc_list)

    return N_ICF, N_ICF_err

### Metllicity Function

def metal(indata, err, iters):
    
    O2 = pn.Atom('O', 2)
    O3 = pn.Atom('O', 3)
    N2 = pn.Atom('N', 2)

    ### Make numpy arrays of all the necessary data for O+ and O++ derivations

    oii3727, oii3727_err = np.array(indata['OII3727_FLUX_CORR']), np.array(indata['OII3727_FLUX_CORR_ERR'])
    hb, hb_err = np.array(indata['HB4861_FLUX_CORR']), np.array(indata['HB4861_FLUX_CORR_ERR'])
    oiii5006, oiii5006_err = np.array(indata['OIII5006_FLUX_CORR']), np.array(indata['OIII5006_FLUX_CORR_ERR'])
    oii7330, oii7330_err = np.array(indata['OII7330_FLUX_CORR_REFIT']), np.array(indata['OII7330_FLUX_CORR_REFIT_ERR'])
    oii7320, oii7320_err = np.array(indata['OII7319_FLUX_CORR_REFIT']), np.array(indata['OII7319_FLUX_CORR_REFIT_ERR'])
    nii6584, nii6584_err = np.array(indata['NII6583_FLUX_CORR']), np.array(indata['NII6583_FLUX_CORR_ERR'])

    oiitemp, oiitemp_err = np.array(indata['OII_TEMP']), np.array(indata['OII_TEMP_ERR'])
    oii_ne, oii_ne_err = np.array(indata['OII_DEN']), np.array(indata['OII_DEN_ERR'])
    niitemp, niitemp_err = np.array(indata['NII_TEMP']), np.array(indata['NII_TEMP_ERR'])
    sii_ne, sii_ne_err = np.array(indata['SII_DEN']), np.array(indata['SII_DEN_ERR'])
    siiitemp, siiitemp_err = np.array(indata['SIII_TEMP']), np.array(indata['SIII_TEMP_ERR'])

    t0, t0_err = np.array(indata['OIII_TEMP_MD23']), np.array(indata['OIII_TEMP_MD23_ERR'])
    tb12, tb12_err = np.array(indata['OIII_TEMP_B12']), np.array(indata['OIII_TEMP_B12_ERR'])

    ### O+ derivation with T([NII]) + [OII]3727 + n([OII])
    O2_NII_3727_OII, O2_NII_3727_OII_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        O2_NII_3727_OII[i], O2_NII_3727_OII_err[i] = ion_function(pyneb_ion=O2, in_flux=oii3727[i], in_temp=niitemp[i], in_den=oii_ne[i], 
                                                            in_eval='L(3726)+L(3729)', in_h=hb[i], mc_iterations=iters, 
                                                            in_flux_err=oii3727_err[i], in_temp_err=niitemp_err[i], 
                                                            in_den_err=oii_ne_err[i], in_h_err=hb_err[i])

    ### O+ derivation with T([NII]) + [OII]3727 + n([SII])
    O2_NII_3727_SII, O2_NII_3727_SII_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        O2_NII_3727_SII[i], O2_NII_3727_SII_err[i] = ion_function(pyneb_ion=O2, in_flux=oii3727[i], in_temp=niitemp[i], in_den=sii_ne[i], 
                                                            in_eval='L(3726)+L(3729)', in_h=hb[i], mc_iterations=iters, 
                                                            in_flux_err=oii3727_err[i], in_temp_err=niitemp_err[i], 
                                                            in_den_err=sii_ne_err[i], in_h_err=hb_err[i])

    ### O+ derivation with T([NII]) + [OII]7320,7330 + n([OII])
    O2_NII_7325_OII, O2_NII_7325_OII_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        O2_NII_7325_OII[i], O2_NII_7325_OII_err[i] = ion_function(pyneb_ion=O2, in_flux=oii7320[i] + oii7330[i], in_temp=niitemp[i], in_den=oii_ne[i], 
                                                            in_eval='L(7319)+L(7320)+L(7330)+L(7331)', in_h=hb[i], mc_iterations=iters, 
                                                            in_flux_err=np.sqrt(oii7320_err[i]**2 + oii7330_err[i]**2), in_temp_err=niitemp_err[i], 
                                                            in_den_err=oii_ne_err[i], in_h_err=hb_err[i])

    ### O+ derivation with T([OII]) + [OII]3727 + n([OII])
    O2_OII_3727_OII, O2_OII_3727_OII_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        O2_OII_3727_OII[i], O2_OII_3727_OII_err[i] = ion_function(pyneb_ion=O2, in_flux=oii3727[i], in_temp=oiitemp[i], in_den=oii_ne[i], 
                                                            in_eval='L(3726)+L(3729)', in_h=hb[i], mc_iterations=iters, 
                                                            in_flux_err=oii3727_err[i], in_temp_err=oiitemp_err[i], 
                                                            in_den_err=oii_ne_err[i], in_h_err=hb_err[i])

    ### O+ derivation with T([NII]) + [OII]7320,7330 + n([SII]) - MUSE only
    O2_NII_7325_SII, O2_NII_7325_SII_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        O2_NII_7325_SII[i], O2_NII_7325_SII_err[i] = ion_function(pyneb_ion=O2, in_flux=oii7320[i] + oii7330[i], in_temp=niitemp[i], in_den=sii_ne[i], 
                                                                  in_eval='L(7319)+L(7320)+L(7330)+L(7331)', in_h=hb[i], mc_iterations=iters, 
                                                                  in_flux_err=np.sqrt(oii7320_err[i]**2 + oii7330_err[i]**2), in_temp_err=niitemp_err[i], 
                                                                  in_den_err=sii_ne_err[i], in_h_err=hb_err[i])

    ### O++ derviation with T([SIII]) + n([OII])
    O3_SIII_OII, O3_SIII_OII_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        O3_SIII_OII[i], O3_SIII_OII_err[i] = ion_function(pyneb_ion=O3, in_flux=oiii5006[i], in_temp=siiitemp[i], in_den=oii_ne[i], 
                                                    in_eval='L(5007)', in_h=hb[i], mc_iterations=iters, 
                                                    in_flux_err=oiii5006_err[i], in_temp_err=siiitemp_err[i], 
                                                    in_den_err=oii_ne_err[i], in_h_err=hb_err[i])

    ### O++ derviation with T([SIII]) + n([SII]) - MUSE only quantities
    O3_SIII_SII, O3_SIII_SII_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        O3_SIII_SII[i], O3_SIII_SII_err[i] = ion_function(pyneb_ion=O3, in_flux=oiii5006[i], in_temp=siiitemp[i], in_den=sii_ne[i], 
                                                    in_eval='L(5007)', in_h=hb[i], mc_iterations=iters, 
                                                    in_flux_err=oiii5006_err[i], in_temp_err=siiitemp_err[i], 
                                                    in_den_err=sii_ne_err[i], in_h_err=hb_err[i])

    ### O++ derviation with T[OIII] + n([OII])
    O3_OIII_OII, O3_OIII_OII_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        O3_OIII_OII[i], O3_OIII_OII_err[i] = ion_function(pyneb_ion=O3, in_flux=oiii5006[i], in_temp=tb12[i], in_den=oii_ne[i], 
                                                    in_eval='L(5007)', in_h=hb[i], mc_iterations=iters, 
                                                    in_flux_err=oiii5006_err[i], in_temp_err=tb12_err[i], 
                                                    in_den_err=oii_ne_err[i], in_h_err=hb_err[i])

    ### O++ derviation with T0(O2+) + n([OII])
    O3_T0_OII, O3_T0_OII_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        O3_T0_OII[i], O3_T0_OII_err[i] = ion_function(pyneb_ion=O3, in_flux=oiii5006[i], in_temp=t0[i], in_den=oii_ne[i], 
                                                in_eval='L(5007)', in_h=hb[i], mc_iterations=iters, 
                                                in_flux_err=oiii5006_err[i], in_temp_err=t0_err[i], 
                                                in_den_err=oii_ne_err[i], in_h_err=hb_err[i])

    ### O++ derviation with T0(O2+) + n([SII])
    O3_T0_SII, O3_T0_SII_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        O3_T0_SII[i], O3_T0_SII_err[i] = ion_function(pyneb_ion=O3, in_flux=oiii5006[i], in_temp=t0[i], in_den=sii_ne[i], 
                                                in_eval='L(5007)', in_h=hb[i], mc_iterations=iters, 
                                                in_flux_err=oiii5006_err[i], in_temp_err=t0_err[i], 
                                                in_den_err=sii_ne_err[i], in_h_err=hb_err[i])
    
    ### O/H derivation with O+ (T([OII]) + [OII]3727 + n([OII])) and O++ (T0 + n([OII]))
    OH_T0_OII_OII_OII_3727, OH_T0_OII_OII_OII_3727_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        OH_T0_OII_OII_OII_3727[i], OH_T0_OII_OII_OII_3727_err[i] = O_elem_function(O2_ion=O2_OII_3727_OII[i], O3_ion=O3_T0_OII[i], 
                                               O2_ion_err=O2_OII_3727_OII_err[i], O3_ion_err=O3_T0_OII_err[i])

    ### O/H derivation with O+ (T([NII]) + [OII]3727 + n([OII])) and O++ (T0 + n([OII]))
    OH_T0_OII_NII_OII_3727, OH_T0_OII_NII_OII_3727_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        OH_T0_OII_NII_OII_3727[i], OH_T0_OII_NII_OII_3727_err[i] = O_elem_function(O2_ion=O2_NII_3727_OII[i], O3_ion=O3_T0_OII[i], 
                                                   O2_ion_err=O2_NII_3727_OII_err[i], O3_ion_err=O3_T0_OII_err[i])
    
    ### O/H derivation with O+ (T([NII]) + [OII]7320,7330 + n([SII])) and O++ (T([SIII]) + n([SII]))
    OH_SIII_SII_NII_SII_7325, OH_SIII_SII_NII_SII_7325_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        OH_SIII_SII_NII_SII_7325[i], OH_SIII_SII_NII_SII_7325_err[i] = O_elem_function(O2_ion=O2_NII_7325_SII[i], O3_ion=O3_SIII_SII[i], 
                                                   O2_ion_err=O2_NII_7325_SII_err[i], O3_ion_err=O3_SIII_SII_err[i])

    ### O/H derivation with O+ (T([NII]) + [OII]3727 + n([SII])) and O++ (T0 + n([SII]))
    OH_T0_SII_NII_SII_3727, OH_T0_SII_NII_SII_3727_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        OH_T0_SII_NII_SII_3727[i], OH_T0_SII_NII_SII_3727_err[i] = O_elem_function(O2_ion=O2_NII_3727_SII[i], O3_ion=O3_T0_SII[i], 
                                                   O2_ion_err=O2_NII_3727_SII_err[i], O3_ion_err=O3_T0_SII_err[i])
        
    ### O/H derivation with O+ (T[NII] + [OII]3727 + n([OII])) and O++ (T[SIII] + n([OII]))
    OH_SIII_OII_NII_OII_3727, OH_SIII_OII_NII_OII_3727_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        OH_SIII_OII_NII_OII_3727[i], OH_SIII_OII_NII_OII_3727_err[i] = O_elem_function(O2_ion=O2_NII_3727_OII[i], O3_ion=O3_SIII_OII[i], 
                                                   O2_ion_err=O2_NII_3727_OII_err[i], O3_ion_err=O3_SIII_OII_err[i])
        
    ### N+ derviation with n([SII]) and Te([NII])
    N2_ion_h, N2_ion_h_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        N2_ion_h[i], N2_ion_h_err[i] = ion_function(pyneb_ion=N2, in_flux=nii6584[i], in_temp=niitemp[i], in_den=sii_ne[i], 
                                        in_eval='L(6584)', in_h=hb[i], mc_iterations=iters, 
                                        in_flux_err=nii6584_err[i], in_temp_err=niitemp_err[i], 
                                        in_den_err=sii_ne_err[i], in_h_err=hb_err[i])
    N2_ion_NII_SII = N2_ion_h / O2_NII_3727_SII
    N2_ion_NII_SII_err = N2_ion_NII_SII * np.sqrt((N2_ion_h_err/N2_ion_h)**2 + (O2_NII_3727_SII_err/O2_NII_3727_SII)**2)

    ### N+ derviation with n([SII]) and Te([OII])
    N2_ion_h, N2_ion_h_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        N2_ion_h[i], N2_ion_h_err[i] = ion_function(pyneb_ion=N2, in_flux=nii6584[i], in_temp=oiitemp[i], in_den=sii_ne[i], 
                                        in_eval='L(6584)', in_h=hb[i], mc_iterations=iters, 
                                        in_flux_err=nii6584_err[i], in_temp_err=oiitemp_err[i], 
                                        in_den_err=sii_ne_err[i], in_h_err=hb_err[i])
    N2_ion_OII_SII = N2_ion_h / O2_NII_3727_SII
    N2_ion_OII_SII_err = N2_ion_OII_SII * np.sqrt((N2_ion_h_err/N2_ion_h)**2 + (O2_NII_3727_SII_err/O2_NII_3727_SII)**2)

    ### Derive N elemental abundance with O+ (T([NII]) + [OII]3727 + n([SII])) and O++ (T[SIII] + n([SII]))
    N_ICF, N_ICF_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        N_ICF[i], N_ICF_err[i] = ICF_N(O2_in = O2_NII_3727_SII[i], O3_in = O3_SIII_SII[i], 
                                       O2_in_err = O2_NII_3727_SII_err[i], O3_in_err = O3_SIII_SII_err[i], iterations = iters)
    N_SIII = np.log10(N2_ion_NII_SII * N_ICF)
    N_SIII_err = (1/np.log(10)) * np.sqrt((N_ICF_err/N_ICF)**2 + (N2_ion_NII_SII_err/N2_ion_NII_SII)**2)

    ###Derive N elemental abundance with O+ (T([NII]) + [OII]3727 + n([SII])) and O++ (T0 + n([SII]))
    N_ICF, N_ICF_err = np.zeros(len(indata)), np.zeros(len(indata))
    for i in range(len(indata)):
        N_ICF[i], N_ICF_err[i] = ICF_N(O2_in = O2_NII_3727_SII[i], O3_in = O3_T0_SII[i], 
                                       O2_in_err = O2_NII_3727_SII_err[i], O3_in_err = O3_T0_SII_err[i], iterations = iters)
    N_T0 = np.log10(N2_ion_NII_SII * N_ICF)
    N_T0_err = (1/np.log(10)) * np.sqrt((N_ICF_err/N_ICF)**2 + (N2_ion_NII_SII_err/N2_ion_NII_SII)**2)

    indata.add_columns([O2_NII_3727_OII, O2_NII_3727_OII_err, O2_NII_7325_OII, O2_NII_7325_OII_err, O2_OII_3727_OII, O2_OII_3727_OII_err, O2_NII_3727_SII, O2_NII_3727_SII_err, O2_NII_7325_SII, O2_NII_7325_SII_err,
                        O3_SIII_OII, O3_SIII_OII_err, O3_SIII_SII, O3_SIII_SII_err, O3_T0_SII, O3_T0_SII_err, O3_OIII_OII, O3_OIII_OII_err, O3_T0_OII, O3_T0_OII_err,
                        OH_T0_OII_OII_OII_3727, OH_T0_OII_OII_OII_3727_err, OH_T0_OII_NII_OII_3727, OH_T0_OII_NII_OII_3727_err, OH_SIII_SII_NII_SII_7325, OH_SIII_SII_NII_SII_7325_err, OH_T0_SII_NII_SII_3727, OH_T0_SII_NII_SII_3727_err, OH_SIII_OII_NII_OII_3727, OH_SIII_OII_NII_OII_3727_err,
                        N2_ion_NII_SII, N2_ion_NII_SII_err, N2_ion_OII_SII, N2_ion_OII_SII_err, N_T0, N_T0_err, N_SIII, N_SIII_err],
                        names=('O2_NII_3727_OII', 'O2_NII_3727_OII_ERR', 'O2_NII_7325_OII', 'O2_NII_7325_OII_ERR', 'O2_OII_3727_OII', 'O2_OII_3727_OII_ERR', 'O2_NII_3727_SII', 'O2_NII_3727_SII_ERR', 'O2_NII_7325_SII', 'O2_NII_7325_SII_ERR',
                               'O3_SIII_OII', 'O3_SIII_OII_ERR', 'O3_SIII_SII', 'O3_SIII_SII_ERR', 'O3_T0_SII', 'O3_T0_SII_ERR', 'O3_OIII_OII', 'O3_OIII_OII_ERR', 'O3_T0_OII', 'O3_T0_OII_ERR',
                               'OH_T0_OII_OII_OII_3727', 'OH_T0_OII_OII_OII_3727_ERR', 'OH_T0_OII_NII_OII_3727', 'OH_T0_OII_NII_OII_3727_ERR', 'OH_SIII_SII_NII_SII_7325', 'OH_SIII_SII_NII_SII_7325_ERR', 'OH_T0_SII_NII_SII_3727', 'OH_T0_SII_NII_SII_3727_ERR', 'OH_SIII_OII_NII_OII_3727', 'OH_SIII_OII_NII_OII_3727_ERR',
                               'N2_ABUN_NII', 'N2_ABUN_NII_ERR', 'N2_ABUN_OII', 'N2_ABUN_OII_ERR', 'N_T0', 'N_T0_ERR', 'N_SIII', 'N_SIII_ERR'))

    return indata


### Function for computing the R cal from Pilyugin & Grebel 2016

def rcal(indata, mc):
    rcal_vals = np.zeros(len(indata))

    for i in range(len(indata)):
        N2 = (indata[i]['NII6583_FLUX_CORR'] * (4/3)) / indata[i]['HB4861_FLUX_CORR']
        R2 = indata[i]['OII3727_FLUX_CORR'] / indata[i]['HB4861_FLUX_CORR']
        R3 = (indata[i]['OIII5006_FLUX_CORR'] * (4/3)) / indata[i]['HB4861_FLUX_CORR']
        if np.log10(N2) >= -0.6:
            rcal_vals[i] = 8.589 + 0.022*np.log10(R3/R2) + 0.399*np.log10(N2) + (-0.137 + 0.164*np.log10(R3/R2) + 0.589*np.log10(N2)) * np.log10(R2)
        elif np.log10(N2) < -0.6:
            rcal_vals[i] = 7.932 + 0.944*np.log10(R3/R2) + 0.695*np.log10(N2) + (0.970 - 0.291*np.log10(R3/R2) - 0.019*np.log10(N2)) * np.log10(R2)
        else:
            rcal_vals[i] = np.nan
    
    rcal_errs = np.zeros(len(indata))
    for i in range(len(indata)):
        if np.isnan(rcal_vals[i]):
            rcal_errs[i] = np.nan
            continue
        iter_arr = np.zeros(mc)
        for j in range(mc):
            N2 = (np.random.normal(indata[i]['NII6583_FLUX_CORR'], indata[i]['NII6583_FLUX_CORR_ERR']) * (4/3)) / np.random.normal(indata[i]['HB4861_FLUX_CORR'], indata[i]['HB4861_FLUX_CORR_ERR'])
            R2 = np.random.normal(indata[i]['OII3727_FLUX_CORR'], indata[i]['OII3727_FLUX_CORR_ERR']) / np.random.normal(indata[i]['HB4861_FLUX_CORR'], indata[i]['HB4861_FLUX_CORR_ERR'])
            R3 = (np.random.normal(indata[i]['OIII5006_FLUX_CORR'], indata[i]['OIII5006_FLUX_CORR_ERR']) * (4/3)) / np.random.normal(indata[i]['HB4861_FLUX_CORR'], indata[i]['HB4861_FLUX_CORR_ERR'])
            if np.log10(N2) >= -0.6:
                iter_arr[j] = 8.589 + 0.022*np.log10(R3/R2) + 0.399*np.log10(N2) + (-0.137 + 0.164*np.log10(R3/R2) + 0.589*np.log10(N2)) * np.log10(R2)
            elif np.log10(N2) < -0.6:
                iter_arr[j] = 7.932 + 0.944*np.log10(R3/R2) + 0.695*np.log10(N2) + (0.970 - 0.291*np.log10(R3/R2) - 0.019*np.log10(N2)) * np.log10(R2)
            else:
                iter_arr[j] = np.nan
        rcal_errs[i] = np.nanstd(iter_arr)
    
    indata.add_columns([rcal_vals, rcal_errs], names=['met_rcal', 'met_rcal_err'])

    return indata

### Function for the two-dimensional R and S calibration from Pilyugin & Grebel 2016

def rs2Dcal(indata, mc):
    cal_vals = np.zeros(len(indata))

    for i in range(len(indata)):
        N2 = (indata[i]['NII6583_FLUX_CORR'] * (4/3)) / indata[i]['HB4861_FLUX_CORR']
        R2 = indata[i]['OII3727_FLUX_CORR'] / indata[i]['HB4861_FLUX_CORR']
        R3 = (indata[i]['OIII5006_FLUX_CORR'] * (4/3)) / indata[i]['HB4861_FLUX_CORR']
        S2 = (indata[i]['SII6716_FLUX_CORR'] + indata[i]['SII6730_FLUX_CORR']) / indata[i]['HB4861_FLUX_CORR']

        if np.log10(N2) >= -0.6:
            cal_vals[i] = 8.589 + 0.329*np.log10(N2) + (-0.205 + 0.549*np.log10(N2)) * np.log10(R2)
        elif np.log10(N2) < -0.6:
            cal_vals[i] = 8.445 + 0.699*np.log10(N2) + (-0.253 + 0.217*np.log10(N2)) * np.log10(S2)
        else:
            cal_vals[i] = np.nan
    
    cal_errs = np.zeros(len(indata))
    for i in range(len(indata)):
        if np.isnan(cal_vals[i]):
            cal_errs[i] = np.nan
            continue
        iter_arr = np.zeros(mc)
        for j in range(mc):
            N2 = (np.random.normal(indata[i]['NII6583_FLUX_CORR'], indata[i]['NII6583_FLUX_CORR_ERR']) * (4/3)) / np.random.normal(indata[i]['HB4861_FLUX_CORR'], indata[i]['HB4861_FLUX_CORR_ERR'])
            R2 = np.random.normal(indata[i]['OII3727_FLUX_CORR'], indata[i]['HB4861_FLUX_CORR_ERR']) / np.random.normal(indata[i]['HB4861_FLUX_CORR'], indata[i]['HB4861_FLUX_CORR_ERR'])
            R3 = (np.random.normal(indata[i]['OIII5006_FLUX_CORR'], indata[i]['OIII5006_FLUX_CORR_ERR']) * (4/3)) / np.random.normal(indata[i]['HB4861_FLUX_CORR'], indata[i]['HB4861_FLUX_CORR_ERR'])
            S2 = (np.random.normal(indata[i]['SII6716_FLUX_CORR'], indata[i]['SII6716_FLUX_CORR_ERR']) + np.random.normal(indata[i]['SII6730_FLUX_CORR'], indata[i]['SII6730_FLUX_CORR_ERR'])) / np.random.normal(indata[i]['HB4861_FLUX_CORR'], indata[i]['HB4861_FLUX_CORR_ERR'])

            if np.log10(N2) >= -0.6:
                iter_arr[j] = 8.589 + 0.329*np.log10(N2) + (-0.205 + 0.549*np.log10(N2)) * np.log10(R2)
            elif np.log10(N2) < -0.6:
                iter_arr[j] = 8.445 + 0.699*np.log10(N2) + (-0.253 + 0.217*np.log10(N2)) * np.log10(S2)
            else:
                iter_arr[j] = np.nan
        cal_errs[i] = np.nanstd(iter_arr)
    
    indata.add_columns([cal_vals, cal_errs], names=['met_rs2Dcal', 'met_rs2Dcal_err'])

    return indata

### Write function for KK04 calibration 

def kk04(indata, mc):
    cal_vals = np.zeros(len(indata))

    def logq(met, y):
        return (32.81 -1.153*y**2 + (met)*(-3.396 - 0.025*y + 0.1444*y**2)) / (4.603 - 0.3119*y - 0.163*y**2 + met*(-0.48 + 0.0271*y + 0.02037*y**2))

    for i in range(len(indata)):
        iter_y = np.log10(indata[i]['OIII5006_FLUX_CORR'] / indata[i]['OII3727_FLUX_CORR'])
        ion_choice = np.log10(indata[i]['NII6583_FLUX_CORR'] / indata[i]['OII3727_FLUX_CORR'])
        R23 = np.log10((indata[i]['OIII5006_FLUX_CORR'] * (4/3) + indata[i]['OII3727_FLUX_CORR']) / indata[i]['HB4861_FLUX_CORR'])

        if np.isnan(ion_choice) or np.isnan(R23):
            cal_vals[i] = np.nan 
            continue

        for j in range(5): 
            if j == 0:
                if ion_choice <= -1.2:
                    out_q = logq(8.2, iter_y)
                    cal_vals[i] = 9.40 + 4.65*R23 - 3.17*R23**2 - out_q*(0.272 + 0.547*R23 - 0.513*R23**2)
                elif ion_choice > -1.2:
                    out_q = logq(8.7, iter_y)
                    cal_vals[i] = 9.72 -0.777*R23 - 0.951*R23**2 - 0.072*R23**3 - 0.811*R23**4 - out_q*(0.0737 - 0.0713*R23 - 0.141*R23**2 + 0.0373*R23**3 - 0.058*R23**4)
                else:
                    cal_vals[i] = np.nan
            else: 
                if cal_vals[i] <= 8.4:
                    out_q = logq(cal_vals[i], iter_y)
                    cal_vals[i] = 9.40 + 4.65*R23 - 3.17*R23**2 - out_q*(0.272 + 0.547*R23 - 0.513*R23**2)
                elif cal_vals[i] > 8.4:
                    out_q = logq(cal_vals[i], iter_y)
                    cal_vals[i] = 9.72 -0.777*R23 - 0.951*R23**2 - 0.072*R23**3 - 0.811*R23**4 - out_q*(0.0737 - 0.0713*R23 - 0.141*R23**2 + 0.0373*R23**3 - 0.058*R23**4)
                else:
                    cal_vals[i] = np.nan
    
    cal_errs = np.zeros(len(indata))
    for i in range(len(indata)):
        if np.isnan(cal_vals[i]):
            cal_errs[i] = np.nan
            continue
        iter_arr = np.zeros(mc)
        for k in range(mc):
            iter_y = np.log10(np.random.normal(indata[i]['OIII5006_FLUX_CORR'], indata[i]['OIII5006_FLUX_CORR_ERR']) / np.random.normal(indata[i]['OII3727_FLUX_CORR'], indata[i]['OII3727_FLUX_CORR_ERR']))
            ion_choice = np.log10(np.random.normal(indata[i]['NII6583_FLUX_CORR'], indata[i]['NII6583_FLUX_CORR_ERR']) / np.random.normal(indata[i]['OII3727_FLUX_CORR'], indata[i]['OII3727_FLUX_CORR_ERR']))
            R23 = np.log10((np.random.normal(indata[i]['OIII5006_FLUX_CORR'], indata[i]['OIII5006_FLUX_CORR_ERR']) * (4/3) + np.random.normal(indata[i]['OII3727_FLUX_CORR'], indata[i]['OII3727_FLUX_CORR_ERR'])) / np.random.normal(indata[i]['HB4861_FLUX_CORR'], indata[i]['HB4861_FLUX_CORR_ERR']))

            for j in range(5): 
                if j == 0:
                    if ion_choice <= -1.2:
                        out_q = logq(8.2, iter_y)
                        iter_arr[k] = 9.40 + 4.65*R23 - 3.17*R23**2 - out_q*(0.272 + 0.547*R23 - 0.513*R23**2)
                    elif ion_choice > -1.2:
                        out_q = logq(8.7, iter_y)
                        iter_arr[k] = 9.72 -0.777*R23 - 0.951*R23**2 - 0.072*R23**3 - 0.811*R23**4 - out_q*(0.0737 - 0.0713*R23 - 0.141*R23**2 + 0.0373*R23**3 - 0.058*R23**4)
                    else:
                        iter_arr[k] = np.nan
                else: 
                    if iter_arr[k] <= 8.4:
                        out_q = logq(iter_arr[k], iter_y)
                        iter_arr[k] = 9.40 + 4.65*R23 - 3.17*R23**2 - out_q*(0.272 + 0.547*R23 - 0.513*R23**2)
                    elif iter_arr[k] > 8.4:
                        out_q = logq(iter_arr[k], iter_y)
                        iter_arr[k] = 9.72 -0.777*R23 - 0.951*R23**2 - 0.072*R23**3 - 0.811*R23**4 - out_q*(0.0737 - 0.0713*R23 - 0.141*R23**2 + 0.0373*R23**3 - 0.058*R23**4)
                    else:
                        iter_arr[k] = np.nan
        cal_errs[i] = np.nanstd(iter_arr)
    
    indata.add_columns([cal_vals, cal_errs], names=['met_kk04', 'met_kk04_err'])

    return indata

### Add M91 calibration

def M91(indata, mc):
    cal_vals = np.zeros(len(indata))

    for i in range(len(indata)):
        x = np.log10((indata[i]['OIII5006_FLUX_CORR'] * (4/3) + indata[i]['OII3727_FLUX_CORR']) / indata[i]['HB4861_FLUX_CORR'])
        y = np.log10((indata[i]['OIII5006_FLUX_CORR'] * (4/3)) / indata[i]['OII3727_FLUX_CORR'])
        ion_choice = np.log10(indata[i]['NII6583_FLUX_CORR'] / indata[i]['OII3727_FLUX_CORR'])


        if ion_choice <= -1.2:
            cal_vals[i] = 12 - 4.944 + 0.767*x + 0.602*x**2 - y*(0.29 + 0.332*x - 0.331*x**2)
        elif ion_choice > -1.2:
            cal_vals[i] = 12 - 2.939 - 0.2*x - 0.237*x**2 - 0.305*x**3 -0.0283*x**4 - y*(0.0047 - 0.0221*x - 0.102*x**2 - 0.0817*x**3 - 0.00717*x**4)
        else:
            cal_vals[i] = np.nan
    
    cal_errs = np.zeros(len(indata))
    for i in range(len(indata)):
        iter_arr = np.zeros(mc)
        if np.isnan(cal_vals[i]):
            cal_errs[i] = np.nan 
            continue
        for j in range(mc):
            x = np.log10((np.random.normal(indata[i]['OIII5006_FLUX_CORR'], indata[i]['OIII5006_FLUX_CORR_ERR']) * (4/3) + np.random.normal(indata[i]['OII3727_FLUX_CORR'], indata[i]['OII3727_FLUX_CORR_ERR'])) / np.random.normal(indata[i]['HB4861_FLUX_CORR'], indata[i]['HB4861_FLUX_CORR_ERR']))
            y = np.log10((np.random.normal(indata[i]['OIII5006_FLUX_CORR'], indata[i]['OIII5006_FLUX_CORR_ERR']) * (4/3)) / np.random.normal(indata[i]['OII3727_FLUX_CORR'], indata[i]['OII3727_FLUX_CORR_ERR']))
            ion_choice = np.log10(np.random.normal(indata[i]['NII6583_FLUX_CORR'], indata[i]['NII6583_FLUX_CORR_ERR']) / np.random.normal(indata[i]['OII3727_FLUX_CORR'], indata[i]['OII3727_FLUX_CORR_ERR']))

            if ion_choice <= -1.2:
                iter_arr[j] = 12 - 4.944 + 0.767*x + 0.602*x**2 - y*(0.29 + 0.332*x - 0.331*x**2)
            elif ion_choice > -1.2:
                iter_arr[j] = 12 - 2.939 - 0.2*x - 0.237*x**2 - 0.305*x**3 -0.0283*x**4 - y*(0.0047 - 0.0221*x - 0.102*x**2 - 0.0817*x**3 - 0.00717*x**4)
            else:
                iter_arr[j] = np.nan
        cal_errs[i] = np.nanstd(iter_arr)
    
    indata.add_columns([cal_vals, cal_errs], names=['met_m91', 'met_m91_err'])

    return indata

### Function for p05 calibration

def P05(indata, mc):
    cal_vals = np.zeros(len(indata))

    for i in range(len(indata)):
        r23 = (indata[i]['OIII5006_FLUX_CORR'] * (4/3) + indata[i]['OII3727_FLUX_CORR']) / indata[i]['HB4861_FLUX_CORR']
        P = ((indata[i]['OIII5006_FLUX_CORR'] * (4/3)) / indata[i]['HB4861_FLUX_CORR']) / r23
        ion_choice = np.log10(indata[i]['NII6583_FLUX_CORR'] / indata[i]['OII3727_FLUX_CORR'])


        if ion_choice <= -1.2:
            cal_vals[i] = (r23 + 106.4 + 106.8*P - 3.40*P**2) / (17.72 + 6.60*P + 6.95*P**2 - 0.302**r23)
        elif ion_choice > -1.2:
            cal_vals[i] = (r23 + 726.1 + 842.2*P + 337.5*P**2) / (85.96 + 82.76*P + 43.98*P**2 + 1.793*r23)
        else:
            cal_vals[i] = np.nan
    
    cal_errs = np.zeros(len(indata))
    for i in range(len(indata)):
        if np.isnan(cal_vals[i]):
            cal_errs[i] = np.nan 
            continue
        iter_arr = np.zeros(mc)
        for j in range(mc):
            r23 = (np.random.normal(indata[i]['OIII5006_FLUX_CORR'], indata[i]['OIII5006_FLUX_CORR_ERR']) * (4/3) + np.random.normal(indata[i]['OII3727_FLUX_CORR'], indata[i]['OII3727_FLUX_CORR_ERR'])) / np.random.normal(indata[i]['HB4861_FLUX_CORR'], indata[i]['HB4861_FLUX_CORR_ERR'])
            P = ((np.random.normal(indata[i]['OIII5006_FLUX_CORR'], indata[i]['OIII5006_FLUX_CORR_ERR']) * (4/3)) / np.random.normal(indata[i]['HB4861_FLUX_CORR'], indata[i]['HB4861_FLUX_CORR_ERR'])) / r23
            ion_choice = np.log10(np.random.normal(indata[i]['NII6583_FLUX_CORR'], indata[i]['NII6583_FLUX_CORR_ERR']) / np.random.normal(indata[i]['OII3727_FLUX_CORR'], indata[i]['OII3727_FLUX_CORR_ERR']))


            if ion_choice <= -1.2:
                iter_arr[j] = (r23 + 106.4 + 106.8*P - 3.40*P**2) / (17.72 + 6.60*P + 6.95*P**2 - 0.302**r23)
            elif ion_choice > -1.2:
                iter_arr[j] = (r23 + 726.1 + 842.2*P + 337.5*P**2) / (85.96 + 82.76*P + 43.98*P**2 + 1.793*r23)
            else:
                iter_arr[j] = np.nan
        cal_errs[i] = np.nanstd(iter_arr)
    
    indata.add_columns([cal_vals, cal_errs], names=['met_p05', 'met_p05_err'])

    return indata

### Write a function for the D02 calibration

def D02(indata, mc):
    cal_vals = np.zeros(len(indata))

    for i in range(len(indata)):
        N2 = np.float64(np.log10(indata[i]['NII6583_FLUX_CORR'] / indata[i]['HA6562_FLUX_CORR']) )

        if ~np.isnan(N2):
            cal_vals[i] = 9.12 + 0.73 * N2
        else:
            cal_vals[i] = np.nan
    
    cal_errs = np.zeros(len(indata))
    for i in range(len(indata)):
        if np.isnan(cal_vals[i]):
            cal_errs[i] = np.nan
            continue 
        iter_arr = np.zeros(mc)
        for j in range(mc):
            N2 = np.float64(np.log10(np.random.normal(indata[i]['NII6583_FLUX_CORR'], indata[i]['NII6583_FLUX_CORR_ERR']) / np.random.normal(indata[i]['HA6562_FLUX_CORR'], indata[i]['HA6562_FLUX_CORR_ERR'])) )

            if ~np.isnan(N2):
                iter_arr[j] = 9.12 + 0.73 * N2
            else:
                iter_arr[j] = np.nan
        cal_errs[i] = np.nanstd(iter_arr)
    
    indata.add_columns([cal_vals, cal_errs], names=['met_d02', 'met_d02_err'])

    return indata

### Add M13 O3N2

def m13_o3n2(indata, mc):
    cal_vals = np.zeros(len(indata))

    for i in range(len(indata)):
        O3N2 = np.float64(np.log10((indata[i]['OIII5006_FLUX_CORR'] / indata[i]['HB4861_FLUX_CORR']) * (indata[i]['HA6562_FLUX_CORR'] / indata[i]['NII6583_FLUX_CORR'])) )

        if ~np.isnan(O3N2):
            cal_vals[i] = 8.533 - 0.214 * O3N2
        else:
            cal_vals[i] = np.nan
    
    cal_errs = np.zeros(len(indata))
    for i in range(len(indata)):
        if np.isnan(cal_vals[i]):
            cal_errs[i] = np.nan
            continue
        iter_arr = np.zeros(mc)
        for j in range(mc):
            O3N2 = np.float64(np.log10((np.random.normal(indata[i]['OIII5006_FLUX_CORR'], indata[i]['OIII5006_FLUX_CORR_ERR']) / np.random.normal(indata[i]['HB4861_FLUX_CORR'], indata[i]['HB4861_FLUX_CORR_ERR'])) * (np.random.normal(indata[i]['HA6562_FLUX_CORR'], indata[i]['HA6562_FLUX_CORR_ERR']) / np.random.normal(indata[i]['NII6583_FLUX_CORR'], indata[i]['NII6583_FLUX_CORR_ERR']))) )

            if ~np.isnan(O3N2):
                iter_arr[j] = 8.533 - 0.214 * O3N2
            else:
                iter_arr[j] = np.nan
        cal_errs[i] = np.nanstd(iter_arr)
    
    indata.add_columns([cal_vals, cal_errs], names=['met_m13_o3n2', 'met_m13_o3n2_err'])

    return indata

### Add M13 N2

def m13_n2(indata, mc):
    cal_vals = np.zeros(len(indata))

    for i in range(len(indata)):
        N2 = np.float64(np.log10((indata[i]['NII6583_FLUX_CORR']/indata[i]['HA6562_FLUX_CORR'])) )

        if ~np.isnan(N2):
            cal_vals[i] = 8.743 + 0.462 * N2
        else:
            cal_vals[i] = np.nan
    
    cal_errs = np.zeros(len(indata))
    for i in range(len(indata)):
        if np.isnan(cal_vals[i]):
            cal_errs[i] = np.nan 
            continue
        iter_arr = np.zeros(mc)
        for j in range(mc):
            N2 = np.float64(np.log10((np.random.normal(indata[i]['NII6583_FLUX_CORR'], indata[i]['NII6583_FLUX_CORR_ERR'])/np.random.normal(indata[i]['HA6562_FLUX_CORR'], indata[i]['HA6562_FLUX_CORR_ERR']))) )

            if ~np.isnan(N2):
                iter_arr[j] = 8.743 + 0.462 * N2
            else:
                iter_arr[j] = np.nan
        cal_errs[i] = np.nanstd(iter_arr)
    
    indata.add_columns([cal_vals, cal_errs], names=['met_m13_n2', 'met_m13_n2_err'])

    return indata

### Add D16

def d16(indata, mc):
    cal_vals = np.zeros(len(indata))

    for i in range(len(indata)):

        y = np.float64(np.log10(indata[i]['NII6583_FLUX_CORR'] / (indata[i]['SII6716_FLUX_CORR'] + indata[i]['SII6730_FLUX_CORR'])) + 0.264 * np.log10((indata[i]['NII6583_FLUX_CORR']/indata[i]['HA6562_FLUX_CORR'])))

        if ~np.isnan(y):
            cal_vals[i] = 8.77 + y + 0.45*(y + 0.3)**5
        else:
            cal_vals[i] = np.nan
    
    cal_errs = np.zeros(len(indata))
    for i in range(len(indata)):
        if np.isnan(cal_vals[i]):
            cal_errs[i] = np.nan 
            continue 
        iter_arr = np.zeros(mc)
        for j in range(mc):

            y = np.float64(np.log10(np.random.normal(indata[i]['NII6583_FLUX_CORR'], indata[i]['NII6583_FLUX_CORR_ERR']) / (np.random.normal(indata[i]['SII6716_FLUX_CORR'], indata[i]['SII6716_FLUX_CORR_ERR']) + np.random.normal(indata[i]['SII6730_FLUX_CORR'], indata[i]['SII6730_FLUX_CORR_ERR']))) + 0.264 * np.log10((np.random.normal(indata[i]['NII6583_FLUX_CORR'], indata[i]['NII6583_FLUX_CORR_ERR'])/np.random.normal(indata[i]['HA6562_FLUX_CORR'], indata[i]['HA6562_FLUX_CORR_ERR']))))

            if ~np.isnan(y):
                iter_arr[j] = 8.77 + y + 0.45*(y + 0.3)**5
            else:
                iter_arr[j] = np.nan
        cal_errs[i] = np.nanstd(iter_arr)
    
    indata.add_columns([cal_vals, cal_errs], names=['met_d16', 'met_d16_err'])

    return indata

### Function for MD23 [NII]5755 relation

def md23(indata, mc):
    cal_vals = np.zeros(len(indata))

    for i in range(len(indata)):

        nii = indata[i]['NII_TEMP']

        if type(nii) == np.float64:
            cal_vals[i] = -1.19*(10**-4)*nii + 9.68 
        else:
            cal_vals[i] = np.nan
    
    cal_errs = np.zeros(len(indata))
    for i in range(len(indata)):
        if np.isnan(cal_vals[i]):
            cal_errs[i] = np.nan 
            continue 
        iter_arr = np.zeros(mc)
        for j in range(mc):

            nii = np.random.normal(indata[i]['NII_TEMP'], indata[i]['NII_TEMP_ERR'])

            if ~np.isnan(nii):
                iter_arr[j] = -1.19*(10**-4)*nii + 9.68 
            else:
                iter_arr[j] = np.nan
        cal_errs[i] = np.nanstd(iter_arr)
    
    indata.add_columns([cal_vals, cal_errs], names=['met_md23', 'met_md23_err'])
    #indata['met_md23'] = cal_vals
    #indata['met_md23_err'] = cal_errs

    return indata

### Function for N/O relation from Thurston 1996

def no_t96(indata, mc):

    def nii_temp_func(r23_in):
        return (6065 + 1600*np.log10(r23_in) + 1878*np.log10(r23_in)**2 + 2803*np.log10(r23_in)**3) / 10**4

    cal_vals, cal_errs = np.zeros(len(indata)), np.zeros(len(indata))

    for i in range(len(indata)):

        nii_6584 = indata[i]['NII6583_FLUX_CORR']
        oii_3727 = indata[i]['OII3727_FLUX_CORR']
        oiii_5006 = indata[i]['OIII5006_FLUX_CORR']
        hb = indata[i]['HB4861_FLUX_CORR']
        r23 = (oii_3727 + oiii_5006*4/3) / hb

        nii_6584_err = indata[i]['NII6583_FLUX_CORR_ERR']
        oii_3727_err = indata[i]['OII3727_FLUX_CORR_ERR']
        oiii_5006_err = indata[i]['OIII5006_FLUX_CORR_ERR']
        hb_err = indata[i]['HB4861_FLUX_CORR_ERR']
        r23_err = ((oii_3727 + oiii_5006*4/3 )/ hb) * np.sqrt((np.sqrt((oii_3727_err)**2 + (oiii_5006_err*4/3)**2) / (oii_3727 + oiii_5006*4/3)) ** 2 + (hb_err / hb)**2)


        if np.isnan(r23) or np.isnan(nii_6584) or np.isnan(oii_3727):
            cal_vals[i], cal_errs[i] = np.nan, np.nan
            continue

        cal_vals[i] = np.log10((nii_6584 * 4/3) / oii_3727) + 0.307 - (0.02*np.log10(nii_temp_func(r23))) - (0.726/nii_temp_func(r23))
        mc_list = []
        for j in range(mc):
            mc_list.append(np.log10((np.random.normal(nii_6584, nii_6584_err) * 4/3) / np.random.normal(oii_3727, oii_3727_err)) + 0.307 - (0.02*np.log10(nii_temp_func(np.random.normal(r23, r23_err)))) - (0.726/nii_temp_func(np.random.normal(r23, r23_err))))
        cal_errs[i] = np.nanstd(mc_list)


    indata.add_columns([cal_vals, cal_errs], names=['NO_t96', 'NO_t96_err'])

    return indata