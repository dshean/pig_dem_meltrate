#! /usr/bin/env python

#This is comparison of DAC and AWS inverse barometer 

import sys
import os
import glob
from datetime import datetime, timedelta

import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from lib import iolib
from lib import malib
from lib import timelib 

def get_closest_idx(lat, lon, nc):
    latkey = 'NbLatitudes'
    lonkey = 'NbLongitudes'
    if not latkey in nc.variables:
        latkey = 'lat'
        lonkey = 'lon'
    if not latkey in nc.variables:
        latkey = 'latitude'
        lonkey = 'longitude'
    nclat = nc.variables[latkey][:]
    nclon = nc.variables[lonkey][:]
    if lon < 0:
        lon += 360 
    #y = ((nclat - lat) == 0).nonzero()
    #x = ((nclon - lon)) == 0).nonzero()
    y = np.argmin(np.abs(nclat - lat))
    x = np.argmin(np.abs(nclon - lon))
    print lat, lon
    print nclat[y], nclon[x]
    return y, x

def sample(v, lat_i, lon_i, pad=1):
    if pad > 0: 
        vsamp = v[:,lat_i-pad:lat_i+pad, lon_i-pad:lon_i+pad]
        out = np.median(vsamp.reshape(vsamp.shape[0], vsamp.shape[1]*vsamp.shape[2]), axis=1)
        malib.print_stats(vsamp)
    else:
        out = v[:, lat_i, lon_i]
    return out

def get_dem_list(dem_list_fn):
    dem_fn_list = np.array([os.path.split(line.strip())[-1] for line in open(dem_list_fn, 'r')])
    dem_dt_list = np.array([timelib.fn_getdatetime(fn) for fn in dem_fn_list])
    dem_dt_list_o = timelib.dt2o(dem_dt_list)
    return dem_fn_list, dem_dt_list, dem_dt_list_o

def filter_dem_fn_list(dem_fn_list):
    import re
    #idx = np.array([bool(re.search('atm|lvis|glas',fn)) for fn in dem_fn_list])
    idx = np.array([bool(re.search('glas',fn)) for fn in dem_fn_list])
    dem_fn_list = dem_fn_list[~idx]
    return idx

#Extract timestamps at these dates
#dem_list_fn = '/scr/pig_stack_20151201_tworows_highcount/shelf_clip/20071020_1438_glas_mean-adj_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans-adj_stack_189_fn_list.txt'
#This includes individual GLAS orbits, individual ATM/LVIS flights, all WV+SPIRIT over shelf
#dem_list_fn = '/scr/pig_stack_20160307_tworows_highcount/shelf_clip/20021128_2050_atm_256m-DEM_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans_stack_417_fn_list.txt'
#dem_list_fn = '/scr/pig_stack_20160307_tworows_highcount/full_extent/20021128_2050_atm_256m-DEM_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans_stack_730_fn_list.txt'
#dem_list_fn = sys.argv[1] 

#dem_fn_list, dem_dt_list, dem_dt_list_o = get_dem_list(dem_list_fn)

#ecmwf_fn = '/Volumes/insar3/dshean/ECMWF/PIG_ECMWF_Pmsl_2012-2015.nc'
#ecmwf_fn = '/Volumes/insar3/dshean/ECMWF/PIG_ECMWF_Pmsl_2002-2016.nc'
ecmwf_fn = '/Users/dshean/Documents/UW/Shean_iceshelf_paper/PIG_ECMWF_all_1979-2016_t2m_skt.nc'
ecmwf_nc = netCDF4.Dataset(ecmwf_fn)

lat = -75.0
lon = -101.5

#All dates are days since Jan 1, 1900
refdate=datetime(1900,1,1)
lat_i, lon_i = get_closest_idx(lat, lon, ecmwf_nc) 
ecmwf_dt = [refdate+timedelta(days=i) for i in ecmwf_nc.variables['time'][:]/24.0]
ecmwf_dt_o = timelib.dt2o(ecmwf_dt)

key = 't2m'
ecmwf_Pmsl = ecmwf_nc.variables[key][:] 
#Pmsl = sample(ecmwf_Pmsl, lat_i, lon_i, pad=0)/100.
Pmsl = sample(ecmwf_Pmsl, lat_i, lon_i, pad=0)
Pmslref = np.median(Pmsl)
print "Median Pmsl: %0.2f" % Pmslref
#Note: output will be the expected displacement due to IB
#To remove, need to subtract from observation
Pmsl_IB = (Pmsl - Pmslref)*-0.01
Pmsl_IB = Pmsl - 273.15

lw = 0.5
f, ax = plt.subplots()
ax.axhline(0, color='k', linestyle='--')
#ax.set_ylabel('ECMWF IB correction (m)')
#ax.set_title('PIG Inverse Barometer Effect from ECMWF Mean Sea Level Pressure')
ax.set_ylabel('ECMWF T 2-m (C)')
ax.set_title('PIG shelf 6-hour ERA-Interim temperatures') 
ax.plot_date(ecmwf_dt, Pmsl_IB, marker=None, linestyle='-', linewidth=lw, label='ECMWF Pmsl')
mask = Pmsl_IB > 0
ax.plot_date(np.array(ecmwf_dt)[mask], Pmsl_IB[mask], marker='o', markersize=3, color='r', linestyle='', linewidth=lw, label='ECMWF Pmsl')
#plt.show()

sys.exit()

"""
#This writes out all available 6-hr samples in the time series
#Useful for later interpolations
#Already done, no need to repeat
out = zip(ecmwf_dt_o, Pmsl_IB)
out_fn = 'pig_ecmwf_IB_2002-2016.csv'
with open(out_fn, 'w') as f:
    f.write('date_o,ecmwf_IB\n')
    f.write('\n'.join('%0.6f,%0.2f' % x for x in out))
"""

f = interp1d(ecmwf_dt_o,Pmsl_IB, kind='linear')
#This takes a long time for ~11K points
#f = interp1d(dt_list_o, v_list, kind='cubic')
#xi = dt_list_o[0] + np.arange(0, (dt_list_o[-1] - dt_list_o[0]), 0.1)
xi = dem_dt_list_o
yi = f(xi)

out = zip(dem_fn_list, dem_dt_list, dem_dt_list_o, yi)

out_fn = os.path.splitext(dem_list_fn)[0]+'_IB.csv'
with open(out_fn, 'w') as f:
    f.write('fn,date,date_o,IB\n')
    f.write('\n'.join('%s,%s,%0.6f,%0.2f' % x for x in out))
