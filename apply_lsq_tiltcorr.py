#! /usr/bin/env python

#This applies existing tilt correction csv to stack with same inputs

import sys
import os

from osgeo import gdal
import numpy as np

from lib import iolib
from lib import malib
from lib import geolib

def apply_tilt_map(a, gt, dx, dy, dz):
    x,y = geolib.get_xy_ma(a, gt)
    tilt = dx*x + dy*y + dz 
    out = a - tilt
    return out, tilt 

stack_fn = sys.argv[1]
#coeff = [-4.0853200321014337e-05,-5.1455914113714195e-06,-68.510290024182609]
coeff_fn = '/scr/pig_stack_20160307_tworows_highcount/full_extent/new_workflow/more_testing/20021128_2050_atm_256m-DEM_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans_stack_730_tide_ib_mdt_geoid_removed_nocorr_offset_-3.10m_filt_DG+LVIS+ATM_2009-2016_lsq_tiltcorr.csv'
if len(sys.argv) > 2:
    coeff_fn = sys.argv[2]

s = malib.DEMStack(stack_fn=stack_fn, stats=True, med=True, trend=True)
hdr = 'fn,date,date_o,tiltx,tilty,tiltz'
#stack.fn_list, stack.date_list, stack.date_list_o, tilt[:,0], tilt[:,1], tilt[:,2]
c = np.genfromtxt(coeff_fn, dtype=None, delimiter=',', names=True)
gt = s.gt

from copy import deepcopy
s_tiltcorr = deepcopy(s)
s_tiltcorr.stack_fn = os.path.splitext(stack_fn)[0]+'_lsq_tiltcorr.npz'

for n,fn in enumerate(s.fn_list):
    print n, fn
    idx = (c['fn'] == fn).nonzero()[0]
    if idx.size > 0:
        tilt = list(c[idx][0])
        print s.source[n], tilt[3:]
        #Only update those with nonzero tilt correction
        if tilt[-1] != 0:
            print "Applying tilt correction"
            a = s.ma_stack[n]
            out, t = apply_tilt_map(a, gt, *tilt[3:]) 
            s_tiltcorr.ma_stack[n] = out
            s_tiltcorr.source[n] = s_tiltcorr.source[n]+'_tiltcorr'

s_tiltcorr.compute_stats()
s_tiltcorr.write_stats()
s_tiltcorr.compute_trend()
s_tiltcorr.write_trend()
s_tiltcorr.savestack()
