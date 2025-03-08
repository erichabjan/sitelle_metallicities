import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join, QTable
import astropy.units as u
import sys
import pyneb as pn
from multiprocessing import Pool
import multiprocessing as mp
import math
from astropy.table import vstack
from multiprocessing.pool import ThreadPool

from concurrent.futures import ThreadPoolExecutor

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

infile = open(inter_data_path + f"/{galaxy}_refit+SITELLEfits_data.fits",'rb')
data = Table.read(infile)

### Set parameters

mciters = 10**3
snerr = 0

### PyNeb Diagnostics

diags = pn.Diagnostics()

diags.addDiag('[OII] b3727/b7325', ('O2', '(L(3726)+L(3729))/(L(7319)+L(7320)+L(7330)+L(7331))', 
                                    'RMS([E(3727A+),E(7319A+)*L(7319A+)/(L(7319A+)+L(7330A+)),E(7330A+)*L(7330A+)/(L(7319A+)+L(7330A+))])'))

### PyNeb atomic class information

icf = pn.ICF()
O2 = pn.Atom('O', 2)
O3 = pn.Atom('O', 3)
N2 = pn.Atom('N', 2)
icf_list = ['Ial06_16']
print(icf.getExpression('Ial06_16'))

# Multiprocessing of Temperature and Density Data

if __name__ == "__main__":

    corenum = 32                            #chosen based of the number of cores
    batch = math.ceil(len(data)/corenum)     #batch determines the number of data points in each batched dataset
    datalist = [data[i:i+batch] for i in range(0, len(data), batch)] #make list of batched data
    
    pool = mp.Pool(processes = len(datalist))          #count processes are inititiated
    mplist = [pool.apply_async(af.dentemp, args = (d, snerr, mciters)) for d in datalist] #each batched dataset is assigned to a core 

results = [mplist[i].get() for i in range(len(mplist))]      #Retrieve parallelized results

# Metallicity

### Create multiprocessing objects

if __name__ == "__main__":
    
    pool = mp.Pool(processes = len(results))          #count processes are inititiated
    mplist2 = [pool.apply_async(af.metal, args = (r, snerr, mciters)) for r in results]
    
### Get results from multiprocessing objects

results2 = [mplist2[i].get() for i in range(len(mplist2))]
metdata = vstack(results2)

metlen = len(np.where(~np.isnan(metdata['OH_T0_OII']))[0])
print(f'{metlen} Metallicity measurements')

# Save Metallicity Results

metdata.write(inter_data_path + f'/{galaxy}_physdata_MUSE+SITELLE.fits', overwrite=True)

