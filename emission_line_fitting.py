import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join, QTable
import astropy.units as u
import sys
import pyneb as pn
from astropy.io import fits
from matplotlib import gridspec
from astropy.io import ascii 
import scipy
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy.linalg as la
import multiprocessing as mp
import os
import math
import extinction
from extinction import apply, remove
from orb.fit import fit_lines_in_spectrum
from orb.utils.spectrum import corr2theta, amp_ratio_from_flux_ratio
from orb.core import Lines
from orcs.process import SpectralCube
import gvar
import orb


import sys
sys.path.append("/home/habjan/SITELLE/sitelle_metallicities")

import analysis_functions as af

### Import data

inter_data_path = '/home/habjan/SITELLE/data/data_raw_intermediate'

import argparse

parser = argparse.ArgumentParser(description="Script that accepts a string input.")
parser.add_argument("input_string", type=str, help="The input string to process")
args = parser.parse_args()
galaxy = args.input_string

galveldic = {'NGC4254': 2388 , 'NGC4535': 1954  , 'NGC3351': 775, 'NGC2835': 867, 'NGC0628':651, 'NGC3627':715}
galvel = galveldic[galaxy]
galebv = {'NGC4254': 0.0334 , 'NGC4535': 0.0168 , 'NGC3351': 0.0239, 'NGC2835': 0.0859, 'NGC0628': 0.0607, 'NGC3627':0.0287}
ebv = galebv[galaxy]

hdu1 = fits.open(inter_data_path + f'/{galaxy}_VorSpectra.fits')
hdu2 = fits.open(inter_data_path + f'/{galaxy}_ppxf-bestfit-emlines.fits')

bestfit_gas = np.ma.MaskedArray( hdu2[1].data["BESTFIT"],
                                mask=hdu2[1].data['BESTFIT']==0)

gas_templ = np.ma.MaskedArray( hdu2[1].data["GAS_BESTFIT"],
                              mask=hdu2[1].data['BESTFIT']==0)

lam = np.exp(hdu1[2].data['LOGLAM'])
log_spec = hdu1[1].data['SPEC']

infile = open(inter_data_path + "/Nebulae_catalogue_v3.fits",'rb')
hdul = Table.read(infile)
data = hdul[hdul['gal_name'] == f'{galaxy}']

infile = inter_data_path + f"/{galaxy}_cube.hdf5"
cube = SpectralCube(infile)

hdul = fits.open(inter_data_path + f'/{galaxy}_SITELLE_Spectra.fits')
sitlam = hdul[0].data
sitspec = hdul[1].data

### Chose a S/N value to cut values

sn = 5
fluxin = log_spec - (bestfit_gas - gas_templ)
mciters = 10**3

### Multiprocessing of the refit functions

if __name__ == "__main__":

    pronum = 16 

    split = pronum/5
    batch = math.ceil(len(data)/split)
    fluxlist = [fluxin[i:i+batch] for i in range(0, len(data), batch)]
    speclist = [sitspec[i:i+batch] for i in range(0, len(data), batch)]
    datalist = [data[i:i+batch] for i in range(0, len(data), batch)]

    paramlist = [(lam, fluxlist[i], sn, datalist[i], galvel, mciters, ebv) for i in range(len(datalist))]
    paramlist5755 = [(lam, fluxlist[i], sn, datalist[i], galvel, mciters, ebv, galaxy) for i in range(len(datalist))]
    sitparamlist = [(datalist[i], speclist[i], sitlam, cube, galvel, ebv, sn, mciters) for i in range(len(datalist))]
    funcs = [af.refit5755, af.refit6312, af.refit7319, af.refit7330, af.fit_3727]
    
    pool = mp.Pool(processes = pronum)          #count processes are inititiated

    list5755 = [pool.apply_async(funcs[0], args = p) for p in paramlist5755]
    list6312 = [pool.apply_async(funcs[1], args = p) for p in paramlist]
    list7319 = [pool.apply_async(funcs[2], args = p) for p in paramlist]
    list7330 = [pool.apply_async(funcs[3], args = p) for p in paramlist]
    list3727 = [pool.apply_async(funcs[4], args = p) for p in sitparamlist]

results5755 = [list5755[i].get() for i in range(len(list5755))]
results6312 = [list6312[i].get() for i in range(len(list6312))]
results7319 = [list7319[i].get() for i in range(len(list7319))]
results7330 = [list7330[i].get() for i in range(len(list7330))]
results3727 = [list3727[i].get() for i in range(len(list3727))]

ref5755 = np.concatenate([results5755[i][0] for i in range(len(list5755))]).astype(np.float64)
ref5755err = np.concatenate([results5755[i][1] for i in range(len(list5755))]).astype(np.float64)
wave5755 = np.concatenate([results5755[i][2] for i in range(len(list5755))]).astype(np.float64)
params5755 = np.concatenate([results5755[i][3] for i in range(len(list5755))]).astype(np.float64)
snr5755 = np.concatenate([results5755[i][4] for i in range(len(list5755))]).astype(np.float64)

ref6312 = np.concatenate([results6312[i][0] for i in range(len(list6312))]).astype(np.float64)
ref6312err = np.concatenate([results6312[i][1] for i in range(len(list6312))]).astype(np.float64)
wave6312 = np.concatenate([results6312[i][2] for i in range(len(list6312))]).astype(np.float64)
params6312 = np.concatenate([results6312[i][3] for i in range(len(list6312))]).astype(np.float64)
snr6312 = np.concatenate([results6312[i][4] for i in range(len(list6312))]).astype(np.float64)

ref7319 = np.concatenate([results7319[i][0] for i in range(len(list7319))]).astype(np.float64)
ref7319err = np.concatenate([results7319[i][1] for i in range(len(list7319))]).astype(np.float64)
wave7319 = np.concatenate([results7319[i][2] for i in range(len(list7319))]).astype(np.float64)
params7319 = np.concatenate([results7319[i][3] for i in range(len(list7319))]).astype(np.float64)
snr7319 = np.concatenate([results7319[i][4] for i in range(len(list7319))]).astype(np.float64)

ref7330 = np.concatenate([results7330[i][0] for i in range(len(list7330))]).astype(np.float64)
ref7330err = np.concatenate([results7330[i][1] for i in range(len(list7330))]).astype(np.float64)
wave7330 = np.concatenate([results7330[i][2] for i in range(len(list7330))]).astype(np.float64)
params7330 = np.concatenate([results7330[i][3] for i in range(len(list7330))]).astype(np.float64)
snr7330 = np.concatenate([results7330[i][4] for i in range(len(list7330))]).astype(np.float64)

ref3727 = np.concatenate([results3727[i][0] for i in range(len(list3727))]).astype(np.float64)
ref3727err = np.concatenate([results3727[i][1] for i in range(len(list3727))]).astype(np.float64)
vel3727 = np.concatenate([results3727[i][2] for i in range(len(list3727))]).astype(np.float64)
vel3727err = np.concatenate([results3727[i][3] for i in range(len(list3727))]).astype(np.float64)
wave3727 = np.concatenate([results3727[i][4] for i in range(len(list3727))]).astype(np.float64)
params3727 = np.concatenate([results3727[i][5] for i in range(len(list3727))]).astype(np.float64)
snr3727 = np.concatenate([results3727[i][6] for i in range(len(list3727))]).astype(np.float64)


print(f'{len(np.where(~np.isnan(ref5755))[0])} [NII]5755 flux measurements, {len(np.where(~np.isnan(ref6312))[0])} [SIII]6312 flux measurements, {len(np.where(~np.isnan(ref7319))[0])} [OII]7319 flux measurements, {len(np.where(~np.isnan(ref7330))[0])} [OII]7330 flux measurements, {len(np.where(~np.isnan(ref3727))[0])} [OII]3727 flux measurements')

### Save a FITS file with the spectra and fit parameters

### SITELLE spectral data and fit parameters
primary_hdu = fits.PrimaryHDU(sitlam)
image_hdu_1 = fits.ImageHDU(sitspec)
image_hdu_2 = fits.ImageHDU(params3727)
image_hdu_3 = fits.ImageHDU(snr3727)

### MUSE spectral data
image_hdu_4 = fits.ImageHDU(lam)
image_hdu_5 = fits.ImageHDU(fluxin.data)

### Auroral line fit parameters
image_hdu_6 = fits.ImageHDU(params5755)
image_hdu_7 = fits.ImageHDU(snr5755)

image_hdu_8 = fits.ImageHDU(params6312)
image_hdu_9 = fits.ImageHDU(snr6312)

image_hdu_10 = fits.ImageHDU(params7319)
image_hdu_11 = fits.ImageHDU(snr7319)

image_hdu_12 = fits.ImageHDU(params7330)
image_hdu_13 = fits.ImageHDU(snr7330)

pdf_fit = fits.HDUList([primary_hdu, image_hdu_1, image_hdu_2, image_hdu_3, image_hdu_4, image_hdu_5, image_hdu_6, image_hdu_7, image_hdu_8,
                        image_hdu_9, image_hdu_10, image_hdu_11, image_hdu_12, image_hdu_13])

pdf_fit.writeto(f'/home/habjan/SITELLE/sandbox_notebooks/pdf_fit_plotting/{galaxy}_spectra_fitting.fits', overwrite=True)

### Compile Refits

data.add_columns([ref5755, ref6312, ref7319, ref7330, ref5755err, ref6312err, ref7319err, ref7330err, ref3727, ref3727err, vel3727, vel3727err], 
                 names=('NII5754_FLUX_REFIT','SIII6312_FLUX_REFIT','OII7319_FLUX_REFIT', 'OII7330_FLUX_REFIT',
                       'NII5754_FLUX_REFIT_ERR','SIII6312_FLUX_REFIT_ERR','OII7319_FLUX_REFIT_ERR', 'OII7330_FLUX_REFIT_ERR',
                       'OII3727_FLUX', 'OII3727_FLUX_ERR', 'OII3727_VEL', 'OII3727_VEL_ERR'))

### Set manually chosen bad fits/spectra equal to a nan

bad_fits_all_dict = {'NGC0628':{'NII5754_FLUX_REFIT': np.array([100, 773, 826]),
                            'SIII6312_FLUX_REFIT': np.array([]),
                            'OII7319_FLUX_REFIT': np.array([]),
                            'OII7330_FLUX_REFIT': np.array([])},
                'NGC2835':{'NII5754_FLUX_REFIT': np.array([53, 116]),
                            'SIII6312_FLUX_REFIT': np.array([58, 83, 377, 777, 793, 800, 1102]),
                            'OII7319_FLUX_REFIT': np.array([26, 80, 333, 885, 907]),
                            'OII7330_FLUX_REFIT': np.array([])},
                'NGC3351':{'NII5754_FLUX_REFIT': np.array([43, 217, 303, 369, 661, 984, 1136]),
                            'SIII6312_FLUX_REFIT': np.array([]),
                            'OII7319_FLUX_REFIT': np.array([41, 185]),
                            'OII7330_FLUX_REFIT': np.array([])},
                'NGC3627':{'NII5754_FLUX_REFIT': np.array([85, 128, 743, 995, 1144]),
                            'SIII6312_FLUX_REFIT': np.array([]),
                            'OII7319_FLUX_REFIT': np.array([1179]),
                            'OII7330_FLUX_REFIT': np.array([1179])},
                'NGC4535':{'NII5754_FLUX_REFIT': np.array([186, 495, 1011]),
                            'SIII6312_FLUX_REFIT': np.array([]),
                            'OII7319_FLUX_REFIT': np.array([855]),
                            'OII7330_FLUX_REFIT': np.array([])},}

bad_fits = bad_fits_all_dict[galaxy]

for i in bad_fits:
    try:
        data[i][bad_fits[i]] = np.nan
    except:
        continue

### Correct for extinction

refitdata = af.corr(data, wave7319, wave7330, wave5755, wave6312, wave3727)

# Wavelength calibration

### Find wavelength of blended [OII]3726,3729 using theoretical intensity relation

OII3726 = Lines().get_line_nm('[OII]3726') * 10
OII3729 = Lines().get_line_nm('[OII]3729') * 10

wave1 = OII3726 + ((OII3729 - OII3726) / 1.4)

### Use the OII Velocities from the 5 > SNR fits above to correct the wavelength axis of SITELLE

### Create arrays of high SNR velocity arrays

ha_arr, ha_err_arr = np.array(refitdata['HA6562_VEL']), np.array(refitdata['HA6562_VEL_ERR'])
nii_arr, nii_err_arr = np.array(refitdata['NII6583_VEL']), np.array(refitdata['NII6583_VEL_ERR'])
oii_arr, oii_err_arr = np.array(refitdata['OII3727_VEL']), np.array(refitdata['OII3727_VEL_ERR'])

vel_ind = np.where(~np.isnan(ha_arr + oii_arr + nii_arr) & (abs(oii_arr)>5*oii_err_arr))[0]

ha_vels, ha_err_vels = ha_arr[vel_ind], ha_err_arr[vel_ind]
nii_vels, nii_err_vels = nii_arr[vel_ind], nii_err_arr[vel_ind]
oii_vels, oii_err_vels = oii_arr[vel_ind], oii_err_arr[vel_ind]

### Speed of light

c = 3 * 10**5

### Calculate offset 

ha_dwave = ha_vels * wave1 / c
oii_dwave = oii_vels * wave1 / c
nii_dwave = nii_vels * wave1 / c

ha_cal = oii_dwave - ha_dwave
nii_cal = oii_dwave - nii_dwave

ha_offset = np.median(ha_cal)
nii_offset = np.median(nii_cal)

### Monte Carlo uncertainties of wavelength calibration

ha_mc = np.zeros(mciters)
nii_mc = np.zeros(mciters)

for i in range(mciters):
    ha_dwave_e = np.random.normal(ha_vels, ha_err_vels) * wave1 / c
    oii_dwave_e = np.random.normal(oii_vels, oii_err_vels) * wave1 / c
    nii_dwave_e = np.random.normal(nii_vels, nii_err_vels) * wave1 / c

    ha_cal_e = oii_dwave_e - ha_dwave_e
    nii_cal_e = oii_dwave_e - nii_dwave_e

    ha_offset_e = np.median(ha_cal_e)
    nii_offset_e = np.median(nii_cal_e)

    if ~np.isnan(ha_offset):
        ha_mc[i] = ha_offset_e
    else:
        ha_mc[i] = np.nan
    if ~np.isnan(nii_offset):
        nii_mc[i] = nii_offset_e
    else:
        nii_mc[i] = np.nan

ha_cal_err = np.nanstd(ha_mc)
nii_cal_err = np.nanstd(nii_mc)

### Create calibrated wavelength axis

new_wave = 1/((1/(sitlam * 10**-8) - nii_offset) * 10**-8)

### Correct OII velocities in catalog

refitdata['OII3727_VEL'] = refitdata['OII3727_VEL'] - ((nii_offset / wave1) * c)

### Import old data and overwrite old wavelength axis with calibrated wavelength

sit_spec = fits.open(inter_data_path  + f'/{galaxy}_SITELLE_Spectra.fits')
cube_fits = fits.open(inter_data_path  + f"/{galaxy}_SITELLE.fits")
cube_fits_mp = fits.open(inter_data_path  + f"/{galaxy}_SITELLE_mp.fits")

sit_spec[0].data = new_wave
cube_fits[1].data = new_wave 
cube_fits_mp[1].data = new_wave

### Save new data cubes and spectra

products_data_path = '/home/habjan/SITELLE/data/data_products'

refitdata.write(inter_data_path  + f'/{galaxy}_refit+SITELLEfits_data.fits', 
                overwrite=os.path.exists(inter_data_path  + f'/{galaxy}_refit+SITELLEfits_data.fits'))

sit_spec.writeto(products_data_path + f'/{galaxy}_SITELLE_Spectra.fits', 
                 overwrite=os.path.exists(products_data_path + f'/{galaxy}_SITELLE_Spectra.fits'))

cube_fits.writeto(products_data_path + f"/{galaxy}_SITELLE.fits", 
                  overwrite=os.path.exists(products_data_path + f"/{galaxy}_SITELLE.fits"), 
                  output_verify='fix')

cube_fits_mp.writeto(products_data_path + f"/{galaxy}_SITELLE_mp.fits", 
                     overwrite=os.path.exists(products_data_path + f"/{galaxy}_SITELLE_mp.fits"))

print('Emission Line fitting ran successfully.')