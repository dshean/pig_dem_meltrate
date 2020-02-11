#! /usr/bin/env python

import sys
import os
import re

from datetime import datetime

import numpy as np

from lib import iolib
from lib import malib
from lib import timelib

if len(sys.argv) != 3:
    sys.exit('Must specify input stack and mode')

mode = None
#Set this to output the inverse of the specified filter
inv = False
#Set this to output both the pass and stop 
both = False

stack_fn = sys.argv[1]
mode = sys.argv[2]

#Load stack
#s = malib.DEMStack(stack_fn=stack_fn, stats=True, trend=True, med=True)
s = malib.DEMStack(stack_fn=stack_fn, stats=False, trend=False, med=False, datestack=False)

#For bed
#s = malib.DEMStack(stack_fn=stack_fn, trend=False, stats=False, datestack=False, med=False)
#Pull out Dutrieux bed
#idx = [3,]

#Use default output filename
#Note: if out_stack_fn is not specified, a new stack fn will be computed with updated min/max dt and count
out_fn = None
#out_fn = os.path.splitext(s.stack_fn)[0]+'_filt.npz'

if mode == 'pig_pre_tiltcorr':
    #Isolate DG (both stereo and mono, trans and nocorr) and altimetry
    idx = np.array([bool(re.search('DG|LVIS|ATM', i)) for i in s.source])
    #Limit to 2009-present, when dh/dt is ~constant
    #LVIS on 10/20/2009 should be first
    min_dt = datetime(2009,1,1)
    max_dt = datetime(2016,1,1)
    idx *= (s.date_list >= min_dt) & (s.date_list <= max_dt) 
    both = True
    out_fn = os.path.splitext(s.stack_fn)[0]+'_filt_DG+LVIS+ATM_2009-2016.npz'
elif mode == 'remove_nocorr':
    #Isoalate remaining nocorr after tiltcorr
    #Source should have tiltcorr appended for all that were adjusted
    idx = np.array([bool(re.search('nocorr$', i)) for i in s.source])
    both = True
    out_fn = os.path.splitext(s.stack_fn)[0]+'_filt_nocorr.npz'
elif mode == 'remove_mono':
    #Isoalate remaining nocorr after tiltcorr
    #Source should have tiltcorr appended for all that were adjusted
    idx = np.array([bool(re.search('mono', i)) for i in s.source])
    both = True
    out_fn = os.path.splitext(s.stack_fn)[0]+'_filt_mono.npz'
elif mode == 'pig_shelf_dt':
    min_dt = datetime(2008,1,1)
    max_dt = datetime(2015,6,1)
    idx = np.array((s.date_list >= min_dt) & (s.date_list <= max_dt))
    both = True
    out_fn = os.path.splitext(s.stack_fn)[0]+'_filt_20080101-20150601.npz'
elif mode == 'pig_gps_dt':
    min_dt = datetime(2011,8,1)
    max_dt = datetime(2014,4,1)
    idx = np.array((s.date_list >= min_dt) & (s.date_list <= max_dt))
    both = True
    out_fn = os.path.splitext(s.stack_fn)[0]+'_filt_20110801-20140401.npz'
#Pull out grids that cover entire GPS area
elif mode == 'pig_gps_test':
    #good_dt = [datetime(2011,12,31).date(), datetime(2012,10,23).date()]
    mincount = 20000
    good_dt = [datetime(2012,1,14).date(), datetime(2012,2,2).date(), datetime(2012,10,23).date(), datetime(2012,11,11).date(), datetime(2013,9,24).date()]
    idx = np.array([d.date() in good_dt for d in s.date_list])
    good = s.ma_stack[idx]
    count = good.reshape(good.shape[0], good.shape[1]*good.shape[2]).count(axis=1)
    idx[idx] *= (count > mincount)
    both = False 
    out_fn = os.path.splitext(s.stack_fn)[0]+'_filt_gpstest.npz'
else: 
    #Sample index creation for subset
    #idx = [bool(re.search('ATM|LVIS|GLAS|DG_stereo$|DG_mono$', i)) for i in s.source]
    #idx = [bool(re.search('ATM|LVIS|GLAS|SPIRIT|DG_stereo$|DG_mono$', i)) for i in s.source]
    #idx = [bool(re.search('ATM|LVIS|GLAS|SPIRIT', i)) for i in s.source]
    #idx = [bool(re.search('DG_stereo$', i)) for i in s.source]
    #idx = [bool(re.search('mono', i)) for i in s.source]
    #idx = [bool(re.search('DG_stereo_nocorr|DG_mono', i)) for i in s.source]
    #idx = [bool(re.search('tiltcorr|medcorr', i)) for i in s.source]
    #idx = [bool(re.search('aspbias', i)) for i in s.source]
    #idx = [bool(re.search('DG_stereo_nocorr|DG_mono_nocorr', i)) for i in s.source]
    #idx = [bool(re.search('DG_stereo_nocorr|DG_mono_nocorr', i)) for i in s.source]
    #idx = [bool(re.search('SPIRIT', i)) for i in s.source]
    #idx = [bool(re.search('DG|SPIRIT|TDM|GLISTIN', i)) for i in s.source]
    #idx = [bool(re.search('TSX', i)) for i in s.source]
    #idx = [bool(re.search('DG_stereo$|LVIS|ATM', i)) for i in s.source]

    #Late season waves
    #late_2015 = [datetime(2015, 8, 22, 15, 43), datetime(2015, 8, 5, 16, 11), datetime(2015, 7, 31, 15, 15), datetime(2015, 7, 19, 17, 12), datetime(2015, 7, 5, 0, 41)]
    #late_2014 = [datetime(2014, 8, 7, 15, 50),  datetime(2014, 7, 4, 15, 46), datetime(2014, 6, 23, 15, 8)]
    #late_2013 = [datetime(2013, 9, 29, 15, 10), datetime(2013, 9, 8, 14, 41), datetime(2013, 8, 22, 15, 8), datetime(2013, 7, 5, 1, 6), datetime(2013, 6, 17, 15, 42)]
    #late_2012 = [datetime(2012, 10, 31, 20, 32), datetime(2012, 9, 1, 20, 24), datetime(2012, 8, 26, 15, 14), datetime(2012, 8, 16, 15, 47), datetime(2012, 7, 20, 16, 2)]
    #late_2011 = [datetime(2011, 9, 27, 15, 31), datetime(2011, 9, 14, 15, 37), datetime(2011, 9, 10, 15, 31), datetime(2011, 8, 14, 16, 9), datetime(2011, 8, 6, 16, 23), datetime(2011, 8, 2, 20, 24), datetime(2011, 7, 30, 15, 20)]
    #late_season_dt = late_2015 + late_2014 + late_2013 + late_2012 + late_2011
    #idx = [d in late_season_dt for d in s.date_list]

    """
    #Approximate min/max DEMs for each year
    mm2015 = [datetime(2015, 8, 22, 15, 43), datetime(2015, 4, 25, 17, 7)]
    mm2014 = [datetime(2014, 8, 7, 15, 50), datetime(2014, 4, 19, 15, 34)]
    mm2013 = [datetime(2013, 9, 29, 15, 10), datetime(2013, 3, 17, 17, 8)]
    mm2012 = [datetime(2012, 8, 26, 15, 14), datetime(2012, 4, 9, 15, 43)]
    mm2011 = [datetime(2011, 9, 27, 15, 31), datetime(2011, 4, 5, 15, 47)]
    #Second is ATM
    #Note: these were datetime for full jak catchment, not lower40km
    #mm2010 = [datetime(2010, 8, 17, 15, 15), datetime(2010, 5, 14, 12, 48)]
    mm2010 = [datetime(2010, 8, 17, 15, 15), datetime(2010, 5, 15, 18, 52)]
    #Second is Moller
    mm2009 = [datetime(2009, 8, 6, 15, 14), datetime(2009, 5, 6, 0, 0)]
    #Second is ATM
    #mm2009 = [datetime(2009, 8, 6, 15, 14), datetime(2009, 4, 29, 8, 32)]
    #mm2009 = [datetime(2009, 8, 6, 15, 14), datetime(2009, 4, 28, 15, 03)]
    mm2008 = [datetime(2008, 8, 2, 15, 5), datetime(2008, 6, 4, 15, 27)]
    #First is SPIRIT, second ATM
    #mm2007 = [datetime(2007, 8, 4, 15, 9), datetime(2007, 5, 10, 14, 26)]
    mm2007 = [datetime(2007, 8, 4, 15, 9), datetime(2007, 5, 10, 14, 22)]
    #Both are ATM
    #mm2007 = [datetime(2007, 9, 19, 18, 6), datetime(2007, 5, 10, 14, 26)]
    #mm2007 = [datetime(2007, 9, 20, 17, 8), datetime(2007, 5, 10, 14, 22)]
    #Just max ATM
    #mm2006 = [datetime(2006, 5, 27, 19, 41),]
    #mm2005 = [datetime(2005, 5, 15, 15, 27),]
    #mm2003 = [datetime(2003, 5, 11, 18, 59),]
    mm2006 = [datetime(2006, 5, 27, 11, 35),]
    mm2005 = [datetime(2005, 5, 15, 9, 45),]
    mm2003 = [datetime(2003, 5, 11, 13, 15),]
    mm = mm2015 + mm2014 + mm2013 + mm2012 + mm2011 + mm2010 + mm2009 + mm2008 + mm2007 + mm2006 + mm2005 + mm2003
    idx = [d in mm for d in s.date_list]
    """

    #Pull out end of summer data
    min_rel_dt = (6, 1)
    max_rel_dt = (12, 31)
    #Pull out spring data
    #min_rel_dt = (1, 1)
    #min_rel_dt = (3, 1)
    #max_rel_dt = (5, 31)
    #max_rel_dt = (6, 1)
    #idx = timelib.rel_dt_list_idx(s.date_list, min_rel_dt, max_rel_dt)

    """
    dt_list = [ \
    (datetime(2002,6,1), datetime(2003,5,31)), \
    (datetime(2003,6,1), datetime(2004,5,31)), \
    (datetime(2004,6,1), datetime(2005,5,31)), \
    (datetime(2005,6,1), datetime(2006,5,31)), \
    (datetime(2006,6,1), datetime(2007,5,31)), \
    (datetime(2007,6,1), datetime(2008,5,31)), \
    (datetime(2008,6,1), datetime(2009,5,31)), \
    (datetime(2009,6,1), datetime(2010,5,31)), \
    (datetime(2010,6,1), datetime(2011,5,31)), \
    (datetime(2011,6,1), datetime(2012,5,31)), \
    (datetime(2012,6,1), datetime(2013,5,31)), \
    (datetime(2013,6,1), datetime(2014,5,31)), \
    (datetime(2014,6,1), datetime(2015,5,31))]

    for min_dt,max_dt in dt_list:
        idx = (s.date_list >= min_dt) & (s.date_list <= max_dt) 
        s_sub = malib.get_stack_subset(s, idx, copy=True, save=True)
    """

    #min_dt = datetime(2007,6,1)
    min_dt = datetime(2007,1,1)
    #max_dt = s.date_list.compressed()[-1]
    #min_dt = datetime(2000,1,1)
    #min_dt = datetime(2009,6,1)
    #max_dt = datetime(2010,6,1)
    #min_dt = datetime(2008,6,1)
    #min_dt = datetime(2012,6,1)
    max_dt = datetime(2016,1,1)
    #max_dt = datetime(2015,12,31)
    #max_dt = datetime(2011,6,1)
    #idx *= (s.date_list >= min_dt) & (s.date_list <= max_dt) 

#Make sure idx is array
idx = np.array(idx)

#Invert
if both:
    if np.any(idx):
        s_sub = malib.get_stack_subset(s, idx, out_stack_fn=out_fn, copy=True, save=True)
    if np.any(~idx):
        if out_fn is not None:
            out_fn = os.path.splitext(out_fn)[0]+'inv.npz'
        s_sub = malib.get_stack_subset(s, ~idx, out_stack_fn=out_fn, copy=True, save=True)
else:
    if inv:
        idx = ~idx
        if out_fn is not None:
            out_fn = os.path.splitext(out_fn)[0]+'inv.npz'
    if np.any(idx):
        s_sub = malib.get_stack_subset(s, idx, out_stack_fn=out_fn, copy=True, save=True)
