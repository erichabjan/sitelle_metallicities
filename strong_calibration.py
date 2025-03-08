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
import seaborn as sns
import corner

import sys
sys.path.append("/home/habjan/SITELLE/sitelle_metallicities")
import analysis_functions as af

### Import Data

galdic = {1:'NGC4254', 2:'NGC4535', 3:'NGC3351', 4:'NGC2835', 5:'NGC0628', 6:'NGC3627'}  ## There is no SITELLE data for NGC 4254
#galaxy = galdic[1]
#galveldic = {'NGC4254': 2388 , 'NGC4535': 1954  , 'NGC3351': 775, 'NGC2835': 867, 'NGC0628':651, 'NGC3627':715}
#galvel = galveldic[galaxy]
#galebv = {'NGC4254': 0.0334 , 'NGC4535': 0.0168 , 'NGC3351': 0.0239, 'NGC2835': 0.0859, 'NGC0628': 0.0607, 'NGC3627':0.0287}
#ebv = galebv[galaxy]

inter_data_path = '/home/habjan/SITELLE/data/data_raw_intermediate'

for i in range(2, len(galdic) + 1):
    if i == 2:
        galaxy = galdic[i]
        infile = open(inter_data_path + f"/{galaxy}_physdata_MUSE+SITELLE.fits",'rb')
        data = Table.read(infile)
    else: 
        galaxy = galdic[i]
        infile = open(inter_data_path + f"/{galaxy}_physdata_MUSE+SITELLE.fits",'rb')
        data = vstack([data, Table.read(infile)])
        

### Number of Monte Carlo interations

mciters = 1000


### Run Rcal function with mulitprocessing

if __name__ == "__main__":

    corenum = int(mp.cpu_count() / 2)                            #chosen based of the number of cores
    batch = math.ceil(len(data)/corenum)     #batch determines the number of data points in each batched dataset
    datalist = [data[i:i+batch] for i in range(0, len(data), batch)] #make list of batched data
    
    pool = mp.Pool(processes = len(datalist))          #count processes are inititiated
    mplist = [pool.apply_async(af.rcal, args = (d, mciters)) for d in datalist] #each batched dataset is assigned to a core 

results = [mplist[i].get() for i in range(len(mplist))]      #Retrieve parallelized results
rcal_data = vstack(results)

### Run RScal function with multiprcoessing

if __name__ == "__main__":

    corenum = int(mp.cpu_count() / 2)                            #chosen based of the number of cores
    batch = math.ceil(len(data)/corenum)     #batch determines the number of data points in each batched dataset
    datalist = [rcal_data[i:i+batch] for i in range(0, len(data), batch)] #make list of batched data
    
    pool = mp.Pool(processes = len(datalist))          #count processes are inititiated
    mplist = [pool.apply_async(af.rs2Dcal, args = (d, mciters)) for d in datalist] #each batched dataset is assigned to a core 

results = [mplist[i].get() for i in range(len(mplist))]      #Retrieve parallelized results
rs_cal_data = vstack(results)

### Run KK04 function with multiprocessing

if __name__ == "__main__":

    corenum = int(mp.cpu_count() / 2)                            #chosen based of the number of cores
    batch = math.ceil(len(data)/corenum)     #batch determines the number of data points in each batched dataset
    datalist = [rs_cal_data[i:i+batch] for i in range(0, len(data), batch)] #make list of batched data
    
    pool = mp.Pool(processes = len(datalist))          #count processes are inititiated
    mplist = [pool.apply_async(af.kk04, args = (d, mciters)) for d in datalist] #each batched dataset is assigned to a core 

results = [mplist[i].get() for i in range(len(mplist))]      #Retrieve parallelized results
kk04_data = vstack(results)

### Multiprocessing of M91 function

if __name__ == "__main__":

    corenum = int(mp.cpu_count() / 2)                            #chosen based of the number of cores
    batch = math.ceil(len(data)/corenum)     #batch determines the number of data points in each batched dataset
    datalist = [kk04_data[i:i+batch] for i in range(0, len(data), batch)] #make list of batched data
    
    pool = mp.Pool(processes = len(datalist))          #count processes are inititiated
    mplist = [pool.apply_async(af.M91, args = (d, mciters)) for d in datalist] #each batched dataset is assigned to a core 

results = [mplist[i].get() for i in range(len(mplist))]      #Retrieve parallelized results
m91_data = vstack(results)

### Multiprocessing of P05 function

if __name__ == "__main__":

    corenum = int(mp.cpu_count() / 2)                            #chosen based of the number of cores
    batch = math.ceil(len(data)/corenum)     #batch determines the number of data points in each batched dataset
    datalist = [m91_data[i:i+batch] for i in range(0, len(data), batch)] #make list of batched data
    
    pool = mp.Pool(processes = len(datalist))          #count processes are inititiated
    mplist = [pool.apply_async(af.P05, args = (d, mciters)) for d in datalist] #each batched dataset is assigned to a core 

results = [mplist[i].get() for i in range(len(mplist))]      #Retrieve parallelized results
p05_data = vstack(results)

### Multiprocessing of D02 function

if __name__ == "__main__":

    corenum = int(mp.cpu_count() / 2)                            #chosen based of the number of cores
    batch = math.ceil(len(data)/corenum)     #batch determines the number of data points in each batched dataset
    datalist = [p05_data[i:i+batch] for i in range(0, len(data), batch)] #make list of batched data
    
    pool = mp.Pool(processes = len(datalist))          #count processes are inititiated
    mplist = [pool.apply_async(af.D02, args = (d, mciters)) for d in datalist] #each batched dataset is assigned to a core 

results = [mplist[i].get() for i in range(len(mplist))]      #Retrieve parallelized results
d02_data = vstack(results)

### Multiprocessing of M13 O3N2 function

if __name__ == "__main__":

    corenum = int(mp.cpu_count() / 2)                            #chosen based of the number of cores
    batch = math.ceil(len(data)/corenum)     #batch determines the number of data points in each batched dataset
    datalist = [d02_data[i:i+batch] for i in range(0, len(data), batch)] #make list of batched data
    
    pool = mp.Pool(processes = len(datalist))          #count processes are inititiated
    mplist = [pool.apply_async(af.m13_o3n2, args = (d, mciters)) for d in datalist] #each batched dataset is assigned to a core 

results = [mplist[i].get() for i in range(len(mplist))]      #Retrieve parallelized results
m13_o3n2_data = vstack(results)

### Multiprocessing for M13 N2 function

if __name__ == "__main__":

    corenum = int(mp.cpu_count() / 2)                            #chosen based of the number of cores
    batch = math.ceil(len(data)/corenum)     #batch determines the number of data points in each batched dataset
    datalist = [m13_o3n2_data[i:i+batch] for i in range(0, len(data), batch)] #make list of batched data
    
    pool = mp.Pool(processes = len(datalist))          #count processes are inititiated
    mplist = [pool.apply_async(af.m13_n2, args = (d, mciters)) for d in datalist] #each batched dataset is assigned to a core 

results = [mplist[i].get() for i in range(len(mplist))]      #Retrieve parallelized results
m13_n2_data = vstack(results)

### Multiprocessing of D16 function

if __name__ == "__main__":

    corenum = int(mp.cpu_count() / 2)                            #chosen based of the number of cores
    batch = math.ceil(len(data)/corenum)     #batch determines the number of data points in each batched dataset
    datalist = [m13_n2_data[i:i+batch] for i in range(0, len(data), batch)] #make list of batched data
    
    pool = mp.Pool(processes = len(datalist))          #count processes are inititiated
    mplist = [pool.apply_async(af.d16, args = (d, mciters)) for d in datalist] #each batched dataset is assigned to a core 

results = [mplist[i].get() for i in range(len(mplist))]      #Retrieve parallelized results
d16_data = vstack(results)

### Multiprocessing for MD23 relation 

if __name__ == "__main__":

    corenum = int(mp.cpu_count() / 2)                            #chosen based of the number of cores
    batch = math.ceil(len(data)/corenum)     #batch determines the number of data points in each batched dataset
    datalist = [d16_data[i:i+batch] for i in range(0, len(data), batch)] #make list of batched data
    
    pool = mp.Pool(processes = len(datalist))          #count processes are inititiated
    mplist = [pool.apply_async(af.md23, args = (d, mciters)) for d in datalist] #each batched dataset is assigned to a core 

results = [mplist[i].get() for i in range(len(mplist))]      #Retrieve parallelized results
md23_data = vstack(results)

### Multiprocessing of NO relation

if __name__ == "__main__":

    corenum = int(mp.cpu_count() / 2)                            #chosen based of the number of cores
    batch = math.ceil(len(data)/corenum)     #batch determines the number of data points in each batched dataset
    datalist = [md23_data[i:i+batch] for i in range(0, len(data), batch)] #make list of batched data
    
    pool = mp.Pool(processes = len(datalist))          #count processes are inititiated
    mplist = [pool.apply_async(af.no_t96, args = (d, mciters)) for d in datalist] #each batched dataset is assigned to a core 

results = [mplist[i].get() for i in range(len(mplist))]      #Retrieve parallelized results
t96_data = vstack(results)

### Save new table with strong line data

prod_data_path = '/home/habjan/SITELLE/data/data_products'

t96_data.write(prod_data_path + f'/strong_line_MUSE+SITELLE.fits', overwrite=True)