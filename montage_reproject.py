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

import sys
sys.path.append("/home/habjan/SITELLE/sitelle_metallicities")

import analysis_functions as af

### Import data

inter_data_path = '/home/habjan/SITELLE/data/data_raw_intermediate'

import argparse

galaxy = 'NGC0628'

galveldic = {'NGC4254': 2388 , 'NGC4535': 1954  , 'NGC3351': 775, 'NGC2835': 867, 'NGC0628':651, 'NGC3627':715}
galvel = galveldic[galaxy]

infile = inter_data_path + f"/{galaxy}_cube.hdf5"
cube = SpectralCube(infile)

hdul = fits.open(inter_data_path + f"/{galaxy}_IMAGE_FOV_Johnson_B_WCS_Pall_mad.fits")
muse_data = NDData(data=hdul['DATA'].data, mask=np.isnan(hdul['DATA'].data), meta=hdul['DATA'].header, wcs=WCS(hdul['DATA'].header)) 

hdul = fits.open(inter_data_path + f"/{galaxy}_SDSS_g.fits")
fits_rband = hdul['DATA']
muse_rband = NDData(data=hdul['DATA'].data, mask=np.isnan(hdul['DATA'].data), meta=hdul['DATA'].header, wcs=WCS(hdul['DATA'].header)) 

hdul = fits.open(inter_data_path + f"/{galaxy}_Johnson_Bcopt.fits")
fits_bband = hdul['DATA']
muse_bband = NDData(data=hdul['DATA'].data, mask=np.isnan(hdul['DATA'].data), meta=hdul['DATA'].header, wcs=WCS(hdul['DATA'].header)) 

hdul = fits.open(inter_data_path + f"/{galaxy}_MAPS.fits")
Halpha = NDData(data=hdul['HA6562_FLUX'].data, mask=np.isnan(hdul['HA6562_FLUX'].data), meta=hdul['HA6562_FLUX'].header, wcs=WCS(hdul['HA6562_FLUX'].header))
Halpha.data[muse_data.data==0] = np.nan

hdul = fits.open(inter_data_path + f"/{galaxy}_nebulae_mask_V2.fits")
mask_fits = fits.open(inter_data_path + f"/{galaxy}_nebulae_mask_V2.fits")
nebulae_mask = NDData(data = hdul[0].data.astype(float), mask=Halpha.mask, meta=hdul[0].header, wcs=WCS(hdul[0].header)) 
nebulae_mask.data[nebulae_mask.data==-1] = np.nan

hdul = fits.open(inter_data_path + f"/{galaxy}_cube.fits")
header = hdul[0].header
wcs = WCS(header,naxis=2)

infile = open(inter_data_path + "/Nebulae_catalogue_v3.fits",'rb')
hdul = Table.read(infile)
musedata = hdul[hdul['gal_name'] == f'{galaxy}']

hdul = fits.open(inter_data_path + f"/{galaxy}_deepframe.fits")
sit_deep = hdul[0]

hdul = fits.open(inter_data_path + f"/{galaxy}_muse_wcs_sources.fits")
muse_sources = Table(hdul[1].data)

hdul = fits.open(inter_data_path + f"/{galaxy}_sitelle_wcs_sources.fits")
sit_sources = Table(hdul[1].data)

coord_dic = {'NGC4535':np.array([188.5851585 , 8.19257]), 'NGC3351':np.array([160.99236896, 11.70541767]), 
             'NGC2835':np.array([139.47045857, -22.35414826]), 'NGC0628':np.array([24.17123567, 15.78081634]),
             'NGC3627':np.array([170.06252, 12.9915])}
zoom_dic = {'NGC4535':np.array([0.05, 0.05]), 'NGC3351':np.array([0.053, 0.053]), 
            'NGC2835':np.array([0.05, 0.05]), 'NGC0628':np.array([0.08, 0.08]),
            'NGC3627':np.array([0.04, 0.085])}

coord_arr = coord_dic[galaxy]

### Make a FITS file from the ORCS spectral cube and import it

montage_path = '/home/habjan/SITELLE/data/Montage_data'
cube.to_fits(montage_path + f"/{galaxy}_SITELLE.fits")
cube_fits = fits.open(montage_path + f"/{galaxy}_SITELLE.fits")


# WCS Correction on SITELLE deep frame

### Update the CRVALs of the SITELLE header to more closely match the WCS of MUSE

muse_world = WCS(mask_fits[0].header).pixel_to_world(muse_sources['xcentroid'], muse_sources['ycentroid'])
muse_ra, muse_dec = muse_world.ra.degree, muse_world.dec.degree

sit_world = WCS(cube_fits[0].header, naxis=2).pixel_to_world(sit_sources['xcentroid'], sit_sources['ycentroid'])
sit_ra, sit_dec = sit_world.ra.degree, sit_world.dec.degree

arcdiff = np.average(np.sqrt((muse_ra - sit_ra)**2 + (muse_dec - sit_dec)**2) * 3600)
print(f'The average difference between datasets is {round(arcdiff, 5)} Degrees')

ra_diff_med = np.median(muse_ra - sit_ra) * 3600
dec_diff_med = np.median(muse_dec - sit_dec) * 3600
print(f'In {galaxy} the median difference in RA is {round(ra_diff_med, 9)}" and in declination is {round(dec_diff_med, 9)}"')

ra_med = cube_fits[0].header['CRVAL1'] + (ra_diff_med/3600)
dec_med = cube_fits[0].header['CRVAL2'] + (dec_diff_med/3600)
cube_fits[0].header['CRVAL1'] = ra_med
cube_fits[0].header['CRVAL2'] = dec_med

wave = cube.params['base_axis'].astype(np.float64)

### Extract a sky background spectrum for the SITELLE cube

skyspec, outpix, mc_percent = af.skyback(cube_fits[0].data, musedata, galaxy, WCS(cube_fits[0].header, naxis=2), coord_arr, galvel, mc=True)


### Sky subtract the entire cube (I know this line looks weird, I am just subtracting the sky background spectrum from every spectrum)

cube_axes = np.transpose(np.subtract(np.transpose(cube_fits[0].data, axes = (1, 2, 0)), skyspec), axes = (2, 0, 1))

# Reprojection of SITELLE data cube

### Make 2d MontagePy header template using the MUSE HII region mask

montage_header = montage_path + f"/{galaxy}_muse_2d_header.hdr"
mGetHdr(f"/home/habjan/SITELLE/data/data_raw_intermediate/{galaxy}_nebulae_mask_V2.fits", montage_header)

### Convert the cube to flux / sr (energy density) using the SITELLE pixel size and save the cube
flux_per_sr = (cube_axes) * (1 / (0.321)**2) * (206265)**2

fits.PrimaryHDU(data = flux_per_sr, header = cube_fits[0].header).writeto(montage_path + f"/{galaxy}_SITELLE.fits", overwrite=True)

### Reproject the cube into MUSE dimensions

montage_dict = mProjectCube(montage_path + f"/{galaxy}_SITELLE.fits",
                            montage_path + f"/{galaxy}_SITELLE_mp.fits",
                            montage_header,
                            energyMode=False, 
                            fullRegion=True)

status = montage_dict['status']

### Stop the script if the reprojection failed

if status == '1':
    print('Montage reprojection failed with the error message:' + str(montage_dict['msg']))
    sys.exit("Montage reprojection failed")

elif status == '0':
    print('Montage reprojection was successful')
    
else: 
    print('No error message')
    sys.exit("Montage returned no error message, which is strange.")
    
### Import MUSE projection of the SITELLE cube

results = fits.open(montage_path + f"/{galaxy}_SITELLE_mp.fits")
sit_data_mp = results[0].data * (1 / (206265)**2) * (0.2)**2
sit_mp_header = results[0].header

### Make fits images for both projections of the SITELLE data

hdu1 = fits.PrimaryHDU(data = cube_axes, header = cube_fits[0].header)
hdu2 = fits.ImageHDU(data = wave)
cube_fits = fits.HDUList([hdu1, hdu2])

hdu1 = fits.PrimaryHDU(data = sit_data_mp, header = sit_mp_header)
hdu2 = fits.ImageHDU(data = wave)
cube_fits_mp = fits.HDUList([hdu1, hdu2])

### Extract all HII region spectra and compile into a single array

spectra = []

for i in range(len(musedata)):
    
    x_pixs, y_pixs = np.where(mask_fits[0].data == i)
    
    specs_i = cube_fits_mp[0].data[:, x_pixs, y_pixs]
    
    spec_i = np.nansum(specs_i, axis=1)
    
    spectra.append(spec_i)

spectra = np.array(spectra)

### Overwrite wavelength axes in fits images and make HDU object for spectra

primary_hdu = fits.PrimaryHDU(wave)
image_hdu = fits.ImageHDU(spectra)
spec_hdu = fits.HDUList([primary_hdu, image_hdu])

### save the original SITELLE data cube, the reprojected SITELLE data cube, and spectra

spec_hdu.writeto(inter_data_path + f'/{galaxy}_SITELLE_Spectra.fits', overwrite=True)
cube_fits.writeto(inter_data_path + f"/{galaxy}_SITELLE.fits", overwrite=True, output_verify='fix')
cube_fits_mp.writeto(inter_data_path + f"/{galaxy}_SITELLE_mp.fits", overwrite=True)

print('Reprojection code ran successfully')