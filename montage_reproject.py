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

parser = argparse.ArgumentParser(description="Script that accepts a string input.")
parser.add_argument("input_string", type=str, help="The input string to process")
args = parser.parse_args()
galaxy = args.input_string

#galaxynum = 4
#galdic = {1:'NGC4254', 2:'NGC4535', 3:'NGC3351', 4:'NGC2835', 5:'NGC0628', 6:'NGC3627'}  #There is no SITELLE data for NGC 4254, NGC 2835 has the best data 
#galaxy = galdic[galaxynum]
#galaxy

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
galaxy


# WCS Correction on SITELLE deep frame

### Update the CRVALs of the SITELLE header to more closely match the WCS of MUSE

muse_world = WCS(mask_fits[0].header).pixel_to_world(muse_sources['xcentroid'], muse_sources['ycentroid'])
muse_ra, muse_dec = muse_world.ra.degree, muse_world.dec.degree

sit_world = cube.get_wcs().pixel_to_world(sit_sources['xcentroid'], sit_sources['ycentroid'])
sit_ra, sit_dec = sit_world.ra.degree, sit_world.dec.degree

arcdiff = np.average(np.sqrt((muse_ra - sit_ra)**2 + (muse_dec - sit_dec)**2) * 3600)
print(f'The average arcsecond difference between datasets is {round(arcdiff, 5)}')

ra_diff_med = np.median(muse_ra - sit_ra) * 3600
dec_diff_med = np.median(muse_dec - sit_dec) * 3600
print(f'In {galaxy} the median difference in RA is {round(ra_diff_med, 7)}" and in declination is {round(dec_diff_med, 7)}"')

ra_med = cube.params['CRVAL1'] + (ra_diff_med/3600)
dec_med = cube.params['CRVAL2'] + (dec_diff_med/3600)
cube.params['CRVAL1'] = ra_med
cube.params['CRVAL2'] = dec_med

sit_deep.header['CRVAL1'] = ra_med
sit_deep.header['CRVAL2'] = dec_med

deep_header = sit_deep.header

### Obtain data from SITELLE ORCS cube

cube_array = cube.get_all_data()
wave = cube.params['base_axis'].astype(np.float64)
cube_axes = np.transpose(cube_array, axes=(1, 0, 2))

### Use function above to extract sky background spectrum

skyspec, outpix, mc_percent = af.skyback(cube_axes, musedata, galaxy, WCS(deep_header), coord_arr, galvel, mc=True)


### Sky subtract the entire cube

cube_axes = np.subtract(cube_axes, skyspec)

# Reprojection of SITELLE data cube

### Make 2d MontagePy header template using the MUSE HII region mask

mGetHdr(f"/home/habjan/jupfiles/data/{galaxy}_nebulae_mask_V2.fits", f"/home/habjan/jupfiles/Montage_data/{galaxy}_2d_header.hdr")

### Use multiprcoessing to speed up the reprojection

if __name__ == "__main__":

    pronum = 16 

    split = pronum
    batch = math.ceil(cube_axes.shape[2]/split)
    datalist = [cube_axes[:, :, i:i+batch] for i in range(0, cube_axes.shape[2], batch)]

    paramlist = [(datalist[i], sit_deep.header, i, galaxy) for i in range(len(datalist))]
    
    pool = mp.Pool(processes = pronum)          #count processes are inititiated

    list_of_data = [pool.apply_async(af.reproject_sit, args = p) for p in paramlist]

results = [list_of_data[i].get() for i in range(len(list_of_data))]

sit_data_mp = np.transpose(np.concatenate([results[i] for i in range(len(results))]), axes=(1, 2, 0))


### Make MUSE WCS header


c1 = fits.Card('SIMPLE', mask_fits[0].header['SIMPLE'], mask_fits[0].header.comments['SIMPLE'])
c2 = fits.Card('BITPIX', mask_fits[0].header['BITPIX'], mask_fits[0].header.comments['BITPIX'])
c3 = fits.Card('NAXIS', 3, mask_fits[0].header.comments['NAXIS'])

c4 = fits.Card('NAXIS1', mask_fits[0].header['NAXIS1'], mask_fits[0].header.comments['NAXIS1'])
c5 = fits.Card('NAXIS2', mask_fits[0].header['NAXIS2'], mask_fits[0].header.comments['NAXIS2'])
c6 = fits.Card('NAXIS3', len(cube.params.base_axis))
c7 = fits.Card('WCSAXES', 3, mask_fits[0].header.comments['WCSAXES'])

c8 = fits.Card('CRPIX1', mask_fits[0].header['CRPIX1'], mask_fits[0].header.comments['CRPIX1'])
c9 = fits.Card('CRPIX2', mask_fits[0].header['CRPIX2'], mask_fits[0].header.comments['CRPIX2'])
c10 = fits.Card('CRPIX3', cube.get_header()['CRPIX3'])

c11 = fits.Card('PC1_1', mask_fits[0].header['PC1_1'], mask_fits[0].header.comments['PC1_1'])
c12 = fits.Card('PC2_2', mask_fits[0].header['PC2_2'], mask_fits[0].header.comments['PC2_2'])

c13 = fits.Card('CDELT1', mask_fits[0].header['CDELT1'], mask_fits[0].header.comments['CDELT1'])
c14 = fits.Card('CDELT2', mask_fits[0].header['CDELT2'], mask_fits[0].header.comments['CDELT2'])
c15 = fits.Card('CDELT3', cube.get_header()['CDELT3'])

c16 = fits.Card('CUNIT1', mask_fits[0].header['CUNIT1'], mask_fits[0].header.comments['CUNIT1'])
c17 = fits.Card('CUNIT2', mask_fits[0].header['CUNIT2'], mask_fits[0].header.comments['CUNIT2'])
c18 = fits.Card('CUNIT3', cube.get_header()['CUNIT3'])

c19 = fits.Card('CTYPE1', mask_fits[0].header['CTYPE1'], mask_fits[0].header.comments['CTYPE1'])
c20 = fits.Card('CTYPE2', mask_fits[0].header['CTYPE2'], mask_fits[0].header.comments['CTYPE2'])
c21 = fits.Card('CTYPE3', 'WAVE')

c22 = fits.Card('CRVAL1', mask_fits[0].header['CRVAL1'], mask_fits[0].header.comments['CRVAL1'])
c23 = fits.Card('CRVAL2', mask_fits[0].header['CRVAL2'], mask_fits[0].header.comments['CRVAL2'])
c24 = fits.Card('CRVAL3', cube.get_header()['CRVAL3'])

c25 = fits.Card('LONPOLE', mask_fits[0].header['LONPOLE'], mask_fits[0].header.comments['LONPOLE'])
c26 = fits.Card('LATPOLE', mask_fits[0].header['LATPOLE'], mask_fits[0].header.comments['LATPOLE'])
c27 = fits.Card('MJDREF', mask_fits[0].header['MJDREF'], mask_fits[0].header.comments['MJDREF'])
c28 = fits.Card('RADESYS', mask_fits[0].header['RADESYS'], mask_fits[0].header.comments['RADESYS'])

in_cards = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, 
            c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28]

cube_header = fits.Header(cards=in_cards)
cube_header


### Make SITELLE WCS header

in_cards_sit = []

header_list = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'object_name', 'ORDER', 'STEP', 'NM_LASER', 'CTYPE1',
               'CRVAL1', 'CUNIT1', 'CRPIX1', 'CDELT1', 'CROTA1', 'CTYPE2', 'CRVAL2', 'CUNIT2', 'CRPIX2', 'CDELT2',
               'CROTA2', 'WAVETYPE', 'CTYPE3', 'CRVAL3', 'CUNIT3', 'CRPIX3', 'CDELT3', 'CROTA3', 'apodization','axis_corr', 
               'WCSAXES', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'RADESYS', 'EQUINOX', 'BP_0_0', 'BP_0_1', 'BP_0_2', 'A_3_0',
               'B_3_0', 'BP_3_0', 'B_1_2', 'B_1_0', 'B_1_1', 'B_2_1', 'B_2_0', 'A_ORDER', 'B_0_3', 'B_0_2', 'B_0_1', 'B_0_0',
               'BP_0_3', 'B_ORDER', 'BP_ORDER', 'BP_1_2', 'AP_ORDER', 'AP_3_0', 'A_1_1', 'A_1_0', 'BP_2_0', 'A_1_2', 'AP_2_1',
               'AP_2_0', 'A_0_0', 'A_0_1', 'A_0_2', 'A_0_3', 'BP_1_1', 'BP_1_0', 'A_2_0', 'A_2_1', 'AP_1_0', 'AP_1_1', 'AP_1_2',
               'BP_2_1', 'AP_0_1', 'AP_0_0', 'AP_0_3', 'AP_0_2', 'BUNIT', 'zpd_index', 'NAXIS3']

for i in header_list:
    if i == 'WCSAXES' or i == 'NAXIS':
        in_cards_sit.append(fits.Card(i, 3))
    else:
        in_cards_sit.append(fits.Card(i, cube.get_header()[i]))

cube_header_sit = fits.Header(cards=in_cards_sit)


### Transpose data to match WCS

sit_data_mp, cube_axes = np.transpose(sit_data_mp, axes=(2, 0, 1)), np.transpose(cube_axes, axes=(2, 0, 1))


### Make fits images for both projections of the SITELLE data

hdu1 = fits.PrimaryHDU(data = cube_axes, header = cube_header_sit)
hdu2 = fits.ImageHDU(data = wave)
cube_fits = fits.HDUList([hdu1, hdu2])

hdu1 = fits.PrimaryHDU(data = sit_data_mp, header = cube_header)
hdu2 = fits.ImageHDU(data = wave)
cube_fits_mp = fits.HDUList([hdu1, hdu2])

### Extract all HII region spectra and compile into a single array

spectra = np.array([np.mean(cube_fits_mp[0].data[:, np.where(mask_fits[0].data == i)[0], np.where(mask_fits[0].data == i)[1]], axis=1) * cube_fits_mp[0].data[:, np.where(mask_fits[0].data == i)[0], np.where(mask_fits[0].data == i)[1]].shape[1] for i in range(len(musedata))])

### Overwrite wavelength axes in fits images and make HDU object for spectra

primary_hdu = fits.PrimaryHDU(wave)
image_hdu = fits.ImageHDU(spectra)
spec_hdu = fits.HDUList([primary_hdu, image_hdu])

### save the original SITELLE data cube, the reprojected SITELLE data cube, and spectra

spec_hdu.writeto(inter_data_path + f'/{galaxy}_SITELLE_Spectra.fits', overwrite=True)
cube_fits.writeto(inter_data_path + f"/{galaxy}_SITELLE.fits", overwrite=True, output_verify='fix')
cube_fits_mp.writeto(inter_data_path + f"/{galaxy}_SITELLE_mp.fits", overwrite=True)

print('Reprojection code ran successfully')