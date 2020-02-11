#! /usr/bin/env python

#David Shean
#12/1/14
#dshean@gmail.com

import os
import sys
import glob
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from mlab.releases import latest_release as matlab

#import matlab.engine
#eng = matlab.engine.start_matlab()

from lib import timelib
from lib import pltlib

#Tidal model setup
model_basedir = '/Volumes/insar5/dshean/Antarctica/tidal_model'
matlab.addpath(model_basedir)
matlab.addpath(model_basedir+'/FUNCTIONS')
model = model_basedir+'/DATA/Model_CATS2008a_opt'

#This is approx PIG shelf center
lat=-75.08865
lon=-100.38043

#dem_list_fn = '/scr/pig_stack_20151201_tworows_highcount/shelf_clip/20021204_1925_atm_mean-adj_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans-adj_stack_203_fn_list.txt'
#dem_list_fn = '/scr/pig_stack_20151201_tworows_highcount/shelf_clip/20071020_1438_glas_mean-adj_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans-adj_stack_189_fn_list.txt'
#dem_list_fn = '/scr/pig_stack_20151201_tworows_highcount/shelf_clip/20071020_1438_glas_mean-adj_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans-adj_stack_189_fn_list.txt'
#This includes individual GLAS orbits, individual ATM/LVIS flights, all WV+SPIRIT over shelf
#dem_list_fn = '/scr/pig_stack_20160307_tworows_highcount/shelf_clip/20021128_2050_atm_256m-DEM_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans_stack_417_fn_list.txt'
#dem_list_fn = '/scr/pig_stack_20160307_tworows_highcount/full_extent/20021128_2050_atm_256m-DEM_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans_stack_730_fn_list.txt'
dem_list_fn = sys.argv[1] 
out_fn = os.path.splitext(dem_list_fn)[0]+'_tides.csv'

fn_list = [os.path.split(line.strip())[-1] for line in open(dem_list_fn, 'r')]
dt_list = np.array([timelib.fn_getdatetime(fn) for fn in fn_list])
dt_list_o = timelib.dt2o(dt_list)
dt_list_mat = [timelib.dt2mat(dt) for dt in dt_list]

print "Running tide model for %i timestamps" % len(dt_list_mat)
tide_a = matlab.tmd_tide_pred('%s' % model, dt_list_mat, lat, lon, 'z')

print "Writing out"
out = zip(fn_list, dt_list, dt_list_o, tide_a)
with open(out_fn, 'w') as f:
    f.write('fn,date,date_o,tide\n')
    f.write('\n'.join('%s,%s,%0.6f,%0.2f' % x for x in out))

#Get continuous tidal record
dt1 = datetime(2002,1,1)
dt2 = datetime(2016,1,1)
dt_int = timedelta(days=1/24.)
dt_list_hr = timelib.dt_range(dt1, dt2, dt_int)
dt_list_hr_o = timelib.dt2o(dt_list_hr)
dt_list_hr_mat = [timelib.dt2mat(dt) for dt in dt_list_hr]

print "Running tide model for %i timestamps" % len(dt_list_hr_mat)
tide_a_hr = matlab.tmd_tide_pred('%s' % model, dt_list_hr_mat, lat, lon, 'z')

print "Writing out"
out = zip(dt_list_hr, dt_list_hr_o, tide_a_hr)
out_fn = os.path.splitext(out_fn)[0]+'_hr.csv'
with open(out_fn, 'w') as f:
    f.write('date,date_o,tide\n')
    f.write('\n'.join('%s,%0.6f,%0.2f' % x for x in out))

import re
#idx = np.array([bool(re.search('atm|lvis|glas',fn)) for fn in fn_list])
idx = np.array([bool(re.search('glas',fn)) for fn in fn_list])

fig, ax = plt.subplots()
ax.plot_date(dt_list[~idx], tide_a[~idx], color='b', label='WV, SPIRIT, ATM, LVIS')
ax.plot_date(dt_list[idx], tide_a[idx], color='b', fillstyle='none', label='GLAS')
pltlib.fmt_date_ax(ax)
pltlib.pad_xaxis(ax)
ax.axhline(0, color='k')
ax.set_ylabel('Tide z (m)')
ax.set_title('CATS2008A Tidal Amplitude for PIG shelf DEMs (%s, %s)' % (lat, lon))
plt.legend()
outfig_fn = os.path.splitext(out_fn)[0]+'.pdf'
plt.savefig(outfig_fn)
#plt.show()
