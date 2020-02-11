#! /usr/bin/env python

#Scale from 0 to 1
#Min ptp for lsq
#Rather than use 1 km v stack, could use same vstack res, then do interp for points at some stride, and map_coord to orig
#Define separate time ranges for interpolation (linearND) and extrapolation (map_coord)
#Moving Window Kriging
#Smoothness constraint in time and space?

#TopoWx has moving window kriging:
#https://github.com/jaredwo/topowx

import os 
import sys

import multiprocessing as mp

from datetime import datetime, timedelta
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.signal
import scipy.interpolate
from sklearn.gaussian_process import GaussianProcess
from scikits import umfpack

from lib import iolib
from lib import malib
from lib import timelib 
from lib import geolib
from lib import pltlib

#Specify the input type
vtype = 'dem'
#This will attempt to load cached files on disk
load_existing = True 
#Shelf mask
clip_to_shelfmask = False
lsq = True
#Set to solve for both dh/dt and tilt
both = True 

#This does the interpolation for a particular time for all points defined by x and y coords
def dto_interp(interpf, x, y, dto):
    return (dto, interpf(x, y, dto.repeat(x.size)).T)

def rangenorm(x, offset=None, scale=None):
    if offset is None:
        offset = x.min()
    if scale is None:
        scale = x.ptp()
    return (x.astype(np.float64) - offset)/scale

#This repeats the first and last array in the stack with a specified time offset
def pad_stack(s, dt_offset=timedelta(365.25)):
    o = s.ma_stack.shape
    new_ma_stack = np.ma.vstack((s.ma_stack[0:1], s.ma_stack, s.ma_stack[-1:]))
    new_date_list = np.ma.hstack((s.date_list[0:1] - dt_offset, s.date_list, s.date_list[-1:] + dt_offset))
    new_date_list_o = timelib.dt2o(new_date_list)
    return new_ma_stack, new_date_list_o

def sample_lsq(x_lsq_ma, ti):
    ti = np.atleast_1d(ti)
    zi = np.ma.masked_all((ti.size, x_lsq_ma.shape[1], x_lsq_ma.shape[2]))
    order = x_lsq_ma.shape[0] - 1
    for n,i in enumerate(ti):
        if order == 1:
            zi[n] = x_lsq_ma[0] + x_lsq_ma[1]*i
        elif order == 2:
            zi[n] = x_lsq_ma[0] + x_lsq_ma[1]*i + x_lsq_ma[2]*i**2
        elif order == 3:
            zi[n] = x_lsq_ma[0] + x_lsq_ma[1]*i + x_lsq_ma[2]*i**2 + x_lsq_ma[3]*i**3
    return zi

def apply_mask(a, m):
    a[:,m] = np.ma.masked

#dem_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/shelf_clip/20071020_1438_glas_mean-adj_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans-adj_stack_189.npz'
#dem_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/tworows_clip_goodtrend/20091207_20150406_DG_SPIRIT/20091207_1452_SPI_11-008_PIG_SPOT_DEM_V1_40m_hae_newfltr-trans_source-DEM-adj_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans-adj_stack_290.npz'
#This is trans only, both stereo and mono
#dem_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/wv_stereo_mono_transonly/20101116_1411_1030010007A8AB00_1030010008813A00-DEM_32m_trans-adj_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans-adj_stack_247.npz'
#dem_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/20021204_1925_atm_mean-adj_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans-adj_stack_320.npz'
#dem_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/20080105_1447_SPI_09-009_PIG_SPOT_DEM_V1_40m_hae_newfltr-trans_source-DEM-adj_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans-adj_stack_295.npz'
#WV only 
#dem_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/orig_stack_all/20101009_1530_1030010007898D00_103001000799BD00-DEM_32m-adj_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans-adj_stack_328.npz'
dem_stack_fn = '/scr/pig_stack_20160307_tworows_highcount/full_extent/DG_LVIS_ATM_only_fortiltcorr/20091020_1718_lvis_256m-DEM_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans_stack_355.npz' 
dem_stack_fn = '/scr/pig_stack_20160307_tworows_highcount/full_extent/DG_LVIS_ATM_only_fortiltcorr/20091020_1718_lvis_256m-DEM_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans_stack_355_nocorr_offset_-3.10m.npz' 
#dem_stack_fn = '/scr/pig_stack_20160307_tworows_highcount/full_extent/DG_LVIS_ATM_only_fortiltcorr/20091020_1718_lvis_256m-DEM_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans_stack_249.npz'
#dem_stack_fn = '/scr/pig_stack_20160307_tworows_highcount/full_extent/DG_LVIS_ATM_only_fortiltcorr/20101116_1411_1030010007A8AB00_1030010008813A00-DEM_32m_trans_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans_stack_222.npz'

#vx_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vx_20150530_1556_22days_20150525_0356-20150605_0356_tsx_mos_track-MayJun15_vx_stack_19_clip.npz'
#vx_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160222_500m/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vx_20150530_1556_22days_20150525_0356-20150605_0356_tsx_mos_track-MayJun15_vx_stack_20_clip.npz'
#vx_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160222_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vx_20150530_1556_22days_20150525_0356-20150605_0356_tsx_mos_track-MayJun15_vx_stack_20_clip.npz'
vx_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vx_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vx_stack_22_clip.npz'

#vy_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160222_500m/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vy_20150530_1556_22days_20150525_0356-20150605_0356_tsx_mos_track-MayJun15_vy_stack_20_clip.npz'
#vy_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160222_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vy_20150530_1556_22days_20150525_0356-20150605_0356_tsx_mos_track-MayJun15_vy_stack_20_clip.npz'
vy_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vy_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vy_stack_22_clip.npz'

#vm_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160222_500m/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vm_20150530_1556_22days_20150525_0356-20150605_0356_tsx_mos_track-MayJun15_vm_stack_20_clip.npz'
#vm_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160222_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vm_20150530_1556_22days_20150525_0356-20150605_0356_tsx_mos_track-MayJun15_vm_stack_20_clip.npz'
vm_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vm_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vm_stack_22_clip.npz'
#This is just full-shelf mosaics
#vm_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vm_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vm_stack_8.npz'

#vm_stack = np.ma.sqrt(vx_stack.ma_stack**2 + vy_stack.ma_stack**2)

#These are for 256 m grids
#test = vm_stack.ma_stack[:,230:250,350:370]
#test = vx_stack.ma_stack[:,190:230,110:150]
#test = vm_stack.ma_stack[:,160:180,230:260]
#These are for 500 m grids
#Near GL
#test = vm_stack.ma_stack[:,20:50,190:220]
#N shelf, shear margin
#test = vm_stack.ma_stack[:,133:160,101:135]

if vtype == 'vm':
    stack_fn = vm_stack_fn
elif vtype == 'vx':
    stack_fn = vx_stack_fn
elif vtype == 'vy':
    stack_fn = vy_stack_fn
elif vtype == 'dem':
    stack_fn = dem_stack_fn

stack_fn = sys.argv[1]

stack = malib.DEMStack(stack_fn=stack_fn, save=False, trend=True, med=True, stats=True)
#Get times of original obs
t = stack.date_list_o.data

if clip_to_shelfmask:
    mask_fn = '/scr/pig_dem_stack/pig_shelf_poly_shean_2011gl.shp'
    outermask_fn = '/scr/pig_stack_20151201_tworows_highcount/pig_vel_mask.shp'
    m_orig = geolib.shp2array(mask_fn, res=stack.res, extent=stack.extent)
    outermask = ~(geolib.shp2array(outermask_fn, res=stack.res, extent=stack.extent))
    m = m_orig

    #Expand shelf mask 3 km upstream for flux gates
    #Should also reduce edge artifacts
    import scipy.ndimage
    it = int(np.ceil(3000./stack.res))
    m = ~(scipy.ndimage.morphology.binary_dilation(~m, iterations=it))

    #Need to make sure to sample mask at appropriate res
    
    apply_mask(stack.ma_stack, m)
    if vtype == 'vm' or vtype == 'vy' or vtype == 'vx':
        apply_mask(stack.ma_stack, outermask)

#This is used frome here on out
test = stack.ma_stack
test_ptp = stack.dt_stack_ptp
test_source = np.array(stack.source)
res = stack.res
gt = np.copy(stack.gt)

if vtype == 'dem':
    stride = 4 
    test = test[:,::stride,::stride]
    test_ptp = test_ptp[::stride,::stride]
    res *= stride 
    print "Using a stride of %i (%0.1f m)" % (stride, res)
    gt[[1,5]] *= stride

def make_test():
    nt = 30 
    shp = (nt,40,60)
    res = 2000
    test = np.ma.masked_all(shp)
    init = np.full((shp[1], shp[2]), 300)
    #Some spatial distribution of dhdt
    dhdt = np.full((shp[1], shp[2]), -2/365.25)
    ti_min = timelib.dt2o(datetime(2008,1,1))
    ti_max = timelib.dt2o(datetime(2015,4,1))
    ti = np.sort(np.random.random(nt) * (ti_max - ti_min))
    for i,t in enumerate(ti):
        test[i] = init + dhdt * ti[i]
  
    tilt_factor = (110000/res)
    i = nt/2
    dx = 3./tilt_factor
    dy = 1./tilt_factor
    dz = -0.5
    print i, dx, dy, dz
    test[i] = apply_tilt(test[i], -dx, -dy, -dz)
    i = 2
    dx = -1.5/tilt_factor
    dy = 0.1/tilt_factor
    dz = 0.4
    print i, dx, dy, dz
    test[i] = apply_tilt(test[i], -dx, -dy, -dz)
    i = -2
    dx = 0.1/tilt_factor
    dy = 0.1/tilt_factor
    dz = 2.5 
    print i, dx, dy, dz
    test[i] = apply_tilt(test[i], -dx, -dy, -dz)
    #x,y,z = malib.get_xyz(test[nt/2])
    #d = (dx*x + dy*y + dz).reshape((shp[1], shp[2]))
    #test[nt/2] += d
    #Need to add t_min back to ti
    return test, (ti + ti_min), res

#def apply_tilt(a, dx, dy, dz):
def apply_tilt(a, xref, yref, dx, dy, dz):
    x,y,z = malib.get_xyz(a)
    #d = dx*x + dy*y + dz 
    #xref = np.mean(x)
    #yref = np.mean(y)
    d = dx*(x-xref) + dy*(y-yref) + dz 
    #return a - d.reshape((a.shape[0], a.shape[1]))
    #out = np.ma.masked_all_like(a)
    #out[y.astype(int),x.astype(int)] = z - d
    tilt = np.ma.masked_all_like(a)
    tilt[y.astype(int),x.astype(int)] = d
    out = a - tilt
    return out, tilt 

def apply_tilt_map(a, gt, dx, dy, dz):
    x,y = geolib.get_xy_ma(a, gt)
    tilt = dx*x + dy*y + dz 
    out = a - tilt
    return out, tilt 

#This is synthetic test case
#test, t, res = make_test()
#test_ptp = np.full_like(test[0], t[-1] - t[0])

print "Orig shape: ", test.shape
#Check to make sure all t have valid data
tcount = test.reshape(test.shape[0], test.shape[1]*test.shape[2]).count(axis=1)
validt_idx = (tcount > 0).nonzero()[0]
test = test[validt_idx]
test_source = test_source[validt_idx]
t = t[validt_idx]
print "New shape: ", test.shape

#Ben suggested running the vx/vy ratio
#Smoothing the scalar magnitude, then recomputing components

"""
#intergrid
#Note: this won't work with different spatial distribution of missing data over time
#Maybe round timestamps to nearest day, then treat time interval as daily with LOTS of missing data
import intergrid
lo=np.array([x.min(), y.min()])
hi=np.array([x.max(), y.max()])
maps = np.array([x, y])
interfunc = intergrid.Intergrid(test, lo=lo, hi=hi)
"""

if not lsq:
    #test_med = malib.nanfill(test, np.nanmedian, axis=0)
    #x,y,dummy = malib.get_xyz(test_med)
    y, x = (test.count(axis=0) > 0).nonzero()
    x = x.astype(int)
    y = y.astype(int)
    #vm_t = test.reshape(test.shape[0], test.shape[1]*test.shape[2])
    vm_t = test[:,y,x]
    vm_t_flat = vm_t.ravel()
    idx = ~np.ma.getmaskarray(vm_t_flat)
    #These are values
    VM = vm_t_flat[idx]

    X = np.tile(x, t.size)[idx]
    Y = np.tile(y, t.size)[idx]
    T = np.repeat(t, x.size)[idx]
    #These are coords
    pts = np.vstack((X,Y,T)).T

    #Normalized versions
    Xn = rangenorm(X)
    Yn = rangenorm(Y) 
    Tn = rangenorm(T) 
    ptsn = np.vstack((Xn,Yn,Tn)).T

#Interpoalte at these times
#ti = np.arange(t.min(), t.max(), 90.0)
ti_min = timelib.dt2o(datetime(2008,1,1))
#ti_min = timelib.dt2o(datetime(2012,1,1))
#ti_max = timelib.dt2o(datetime(2009,1,1))
#ti_max = timelib.dt2o(datetime(2014,9,1))
#ti_max = timelib.dt2o(datetime(2015,4,1))
ti_max = timelib.dt2o(datetime(2015,6,1))
#ti_dt = 120 
#ti_dt = 121.75 
ti_dt = 91.3125
#ti_dt = 365.25 

#Interpolate at these times 
ti = np.arange(ti_min, ti_max, ti_dt)
#ti = t
#Annual - use for discharge analysis?
#ti = timelib.dt2o([datetime(2008,1,1), datetime(2009,1,1), datetime(2010,1,1), datetime(2011,1,1), datetime(2012,1,1), datetime(2013,1,1), datetime(2014,1,1), datetime(2015,1,1)])

if lsq:
    #LSQ Polynomial fitting

    order = 1
    #Unique indices
    min_count = 4 
    min_ptp = 1.5 * 365.25
    #This works for stack of coregistered DEMs
    #max_std = 3 
    max_std = 4
    #Set this for test case
    #max_std = 99 
    min_z = 10 
    #y, x = (test.count(axis=0) >= min_count).nonzero()
    #validmask = ((test.count(axis=0) >= min_count) & (test_ptp >= min_ptp)).data

    print "Applying validmask"
    print "min count: ", min_count
    print "min ptp (days): ", min_ptp
    print "max std (m): ", max_std
    print "min z (m): ", min_z 
    validmask = (test.count(axis=0) >= min_count) & (test_ptp >= min_ptp) & (test.std(axis=0) <= max_std) & (test.mean(axis=0) > min_z)
    validmask = validmask.data

    #This masks main shelf and area upstream of GL
    shp_fn = '/scr/pig_stack_20151201_tworows_highcount/pig_mainshelfmargins_upstreamtrunk_mask_for_tiltcorr.shp'
    m = geolib.shp2array(shp_fn, res=res, extent=stack.extent)
    validmask = validmask & m
  
    #Exclude pixels over shelf 
    if False:
        shp_fn = '/scr/pig_dem_stack/pig_shelf_poly_shean_2011gl.shp'
        m = geolib.shp2array(shp_fn, res=res, extent=stack.extent)
        validmask = validmask & m

    #Mask areas based on preliminary trend and residuals
    if True:
        test_linreg = malib.ma_linreg(test, np.ma.array(timelib.o2dt(t)))
        #This works for stack of coregistered DEMs
        #detrended_thresh = 1.2
        detrended_thresh = 3.0 
        validmask = validmask & (test_linreg[2] < detrended_thresh).data
        trend_thresh = 2.0
        validmask = validmask & (np.abs(test_linreg[0]) < trend_thresh).data
       
        """
        #Now do this at the end, using original stack
        f = malib.iv(test_linreg[0], clim=(-3,1), label='Linear trend (m/yr)')
        f_fn = os.path.splitext(stack_fn)[0]+'_tiltcorr_trend_before.png'
        f.savefig(f_fn, dpi=300)
        f = malib.iv(test_linreg[2], clim=(0,1), label='Detrended std (m)')
        f_fn = os.path.splitext(stack_fn)[0]+'_tiltcorr_detrended_std_before.png'
        f.savefig(f_fn, dpi=300)
        """

    #f = malib.iv(stack.std, clim=(0,5), label='Std (m)')
    #f_fn = os.path.splitext(stack_fn)[0]+'_tiltcorr_std_before.png'
    #f.savefig(f_fn, dpi=300)

    #f = malib.iv(validmask, cmap='gray', clim=(0,1), title='Tilt correction mask')
    f = malib.iv(validmask, cmap='gray', clim=(0,1))
    f_fn = os.path.splitext(stack_fn)[0]+'_tiltcorr_mask.png'
    f.savefig(f_fn, dpi=300)

    #Get valid indices
    y, x = validmask.nonzero()

    #Should probably check to make sure still have nonzero count for each input after validmask 

    #Should scale everything from 0 to 1
    xref = np.mean(x)
    yref = np.mean(y)
    #t_ref = t.min()
    t_ref = t.mean()
    #t_ref = 0

    #Scale x and y with same factor
    #xn = (x - xref)/x.ptp()
    #xn = x - xref
    xn = x
    #yn = (y - yref)/x.ptp()
    #yn = y - yref
    yn = y
    #tn = (t - t_ref)/t.ptp()
    tn = t - t_ref

    #Remove median from every pixel
    #This helps limit the constraint range
    test_ref = np.ma.median(test, axis=0)
    #Can use 0, but intercept will be 0 to 1200 m
    #test_ref = 0
    #test_ref = np.ma.median(test)
    testn = test - test_ref
    #Note: some of the input DEMs may be completely masked
    testn[:,~validmask] = np.ma.masked

    #Then weight

    if both:
        N = (order + 1)*x.size + (3 * t.size)
        M = testn.count()
        A = scipy.sparse.lil_matrix((M,N))
        b = np.zeros(M)
        Aidx = dict(zip(zip(y,x), range(M)))

        #This is minimum width of DEM to allow for tilt correction
        #min_width = 40000/res
        min_width = 40000

        m = 0
        print "Populating matrices: A=(%ix%i), b=(%i)" % (M,N,M)
        xrefa = []
        yrefa = []
        for k in range(t.size):

            #Check source
            #If ATM/LVIS, tilt and offset should be 0, increased weight on dhdt
            #Trans should have increased weight on dhdt, not tilt
            #Not trans - Remove ASP bias
            #Mono reduced weight on tilt

            #print k
            a = testn[k]
            #za should already have ref removed
            xa,ya,za = malib.get_xyz(a) 
            xam,yam = geolib.get_xy_ma(a, gt) 
            xam = xam.compressed()
            yam = yam.compressed()
            zam = a.compressed()
            #Normalize with global offset and scaling
            #xan = (xa - xref)/x.ptp()
            #yan = (ya - yref)/x.ptp()
            #xan = xa
            #yan = ya
            #Note: want to use local offset for each DEM
            #Constraints are set for individual DEMs
            #Proper tilt values can be very high if global offset is used
            xref = np.mean(xam) 
            xrefa.append(xref)
            xan = xam - xref
            yref = np.mean(yam) 
            yrefa.append(yref)
            yan = yam - yref

            if xa.size > 0:
                aidx = np.array([Aidx[myxy] for myxy in zip(ya,xa)])
               
                #Intercept
                A[np.arange(m,m+aidx.size), aidx] = 1 
                #Linear trend
                A[np.arange(m,m+aidx.size), (x.size)+aidx] = tn[k]
                #Add higher order terms
                #Planar coefficients
                #Only add planar coefficients if DG
                #if not bool(re.search('GLAS|LVIS|ATM', test_source[k])):
                if 'DG' in test_source[k]:
                    #This only computes tilt for longer DEMs
                    #if np.sqrt(xa**2+ya**2).ptp() > min_width:
                    dist = np.sqrt(xam**2+yam**2)
                    #Exclude isolated pixels
                    dist_perc = (5,95)
                    dist_ptp = np.percentile(dist, dist_perc).ptp()
                    if dist_ptp > min_width:
                        A[m:m+aidx.size, ((order+1)*x.size)+k*3+0] = xan[:,np.newaxis] 
                        A[m:m+aidx.size, ((order+1)*x.size)+k*3+1] = yan[:,np.newaxis] 
                    A[m:m+aidx.size, ((order+1)*x.size)+k*3+2] = np.ones((xan.size, 1))
                    
                b[m:m+aidx.size] = za
                m += aidx.size
            else:
                print k, t[k], 'No valid pixels remain after masking'

    else:
        #Compute fit coefficients for each pixel
        N = (order + 1) * x.size
        #Number of unique timesteps at all locations
        M = testn.count(axis=0)[y,x].sum() 

        print "Populating matrices: A=(%ix%i), b=(%i)" % (M,N,M)
        A = scipy.sparse.lil_matrix((M,N))
        b = np.zeros(M)
        n = 0
        m = 0

        Aidx = {} 

        for i,j in zip(y,x):
            #Extract values for a single point
            #Should have M timestamps
            b = testn[:,i,j]
            bidx = (~np.ma.getmaskarray(b)).nonzero()[0]
            #if bidx.size >= min_count:
            bvalid = b[bidx].data
            #b0 = np.median(bvalid)
            b[n:n+bidx.size] = bvalid 
            A[n:n+bidx.size, m] = np.ones((bidx.size,1))
            if order == 1:
                A[n:n+bidx.size, m+1:m+1+order] = tn[bidx, np.newaxis]
            elif order == 2:
                A[n:n+bidx.size, m+1:m+1+order] = np.array([tn[bidx], tn[bidx]**2]).T
            elif order == 3:
                A[n:n+bidx.size, m+1:m+1+order] = np.array([tn[bidx], tn[bidx]**2, tn[bidx]**3]).T
            #Aidx[(i,j)] = (slice(n,n+bidx.size),slice(m,m+1+order))
            Aidx[(i,j)] = slice(m,m+1+order)
            n += bidx.size
            m += (order + 1)

    #Regularize 
    if True:
        print "Adding regularization terms"
        
        #These are expected magnitudes of each parameter, in meters or relevant units
        #If using test_ref = 0, this will be ~500-1000
        #Eint = 600
        #If removing median
        Eint = 10 
        Edhdt = 1 
        #Ex = (0.3/1E5)*res
        #Mapped coord
        Ex = (0.2/1E5)
        Ey = Ex/3.0 
        Ez = 0.3 
        print Eint, Edhdt, Ex, Ey, Ez

        E = np.ones(N)
        #These are intercepts
        E[0:x.size] = Eint 
        #dh/dt
        E[x.size:(order+1)*x.size] = Edhdt 
        #x
        E[(order+1)*x.size::3] = Ex 
        #y
        E[(order+1)*x.size+1::3] = Ey 
        #Vertical offset of plane
        E[(order+1)*x.size+2::3] = Ez 

        #Increase tolerance for DEMs without ICP co-registration
        if True:
            import re
            nocorr_idx = np.nonzero([bool(re.search('nocorr', i)) for i in test_source])[0]
            E[(order+1)*x.size+nocorr_idx+2] = 1.0 

        #Mono should have relaxed tilt and offset constraints
        if True:
            import re
            mono_idx = np.nonzero([bool(re.search('mono', i)) for i in test_source])[0]
            E[(order+1)*x.size+mono_idx] *= 2.0 
            E[(order+1)*x.size+mono_idx+1] *= 2.0 
            E[(order+1)*x.size+mono_idx+2] *= 2.0 

        #Invert
        E = 1./E

        #Note: just set these coefficients to 0 initially
        #Just setting these to 0 still allows tilt for altimetry
        if False:
            import re
            notilt_idx = np.nonzero([bool(re.search('GLAS|LVIS|ATM', i)) for i in test_source])[0]
            E[(order+1)*x.size+notilt_idx] = 0.0 
            E[(order+1)*x.size+notilt_idx+1] = 0.0
            E[(order+1)*x.size+notilt_idx+2] = 0.0

        rA = scipy.sparse.diags(E, 0)
        rb = np.zeros(N)
        """
        rA = scipy.sparse.lil_matrix((3*tn.size,N))
        rA[np.arange(0,3*tn.size),np.arange(N-3*tn.size,N)] = 100000
        rb = np.zeros(3*tn.size) 
        """
        
        A = scipy.sparse.vstack([A,rA])
        b = np.hstack([b,rb])

    if True:
        #Add spatial smoothness constraint
        #This is pretty inefficient for large systems
        print "Preparing Smoothness Constraint"
        SC = scipy.sparse.lil_matrix((x.size,N))
        SC_b = np.zeros(x.size)
        #This sets order of constraint, use 1 for the linear term, 0 for the intercept 
        i = 1
        for n,(key,value) in enumerate(Aidx.iteritems()):
            key_up = (key[0]-1,key[1])
            key_down = (key[0]+1,key[1])
            key_left = (key[0],key[1]-1)
            key_right = (key[0],key[1]+1)
            ud = False
            #The [1] index for dictionary values is when multiple values are written
            if (key_up in Aidx) and (key_down in Aidx):
                #SC[n,Aidx[key][1]] = 2
                #SC[n,Aidx[key_up][1]] = -1 
                #SC[n,Aidx[key_down][1]] = -1 
                SC[n,i*x.size+Aidx[key]] = 2
                SC[n,i*x.size+Aidx[key_up]] = -1 
                SC[n,i*x.size+Aidx[key_down]] = -1 
                ud = True
            if (key_left in Aidx) and (key_right in Aidx):
                if ud:
                    #SC[n,Aidx[key][1]] = 4
                    SC[n,i*x.size+Aidx[key]] = 4
                else:
                    #SC[n,Aidx[key][1]] = 2
                    SC[n,i*x.size+Aidx[key]] = 2
                #SC[n,Aidx[key_left][1]] = -1 
                #SC[n,Aidx[key_right][1]] = -1 
                SC[n,i*x.size+Aidx[key_left]] = -1 
                SC[n,i*x.size+Aidx[key_right]] = -1 

        print "Combining matrices"
        A = scipy.sparse.vstack([A,SC])
        b = np.hstack([b,SC_b])

    """
    Tikhonov regularization
    #http://scicomp.stackexchange.com/questions/10671/tikhonov-regularization-in-the-non-negative-least-square-nnls-pythonscipy
    Err = scipy.sparse.eye(M,N)*5.0
    Err_b = np.zeros(M)
    A = scipy.sparse.vstack([A,Err])
    b = np.hstack([b,Err_b])
    """

    print "Converting"
    A = A.tocsr()

    if True:
        f = plt.figure(figsize=(3,7.5))
        plt.spy(A, precision='present', markersize=1, color='k')
        plt.tight_layout()
        f_fn = os.path.splitext(stack_fn)[0]+'_sparse_matrix.png'
        f.savefig(f_fn, dpi=300)

    print "Solving"
    #x_lsq = scipy.sparse.linalg.lsqr(A, b, show=True)

    AT = A.T
    x_lsq = scipy.sparse.linalg.spsolve(AT*A, AT*b, use_umfpack=True)
    #x_lsq = umfpack.spsolve(AT*A, AT*b)
    x_lsq = x_lsq[np.newaxis,:]

    print "Preparing output"
    if both:
        x_lsq_a = x_lsq[0][0:(order+1)*x.size].reshape((order+1, x.size)).T
        x_lsq_ma = np.ma.masked_all((order+1, test.shape[1], test.shape[2]))
        for i in range(order+1):
            x_lsq_ma[i,y,x] = x_lsq_a[:,i]
        #for i in range(order+1):
        #    for coord,j in Aidx.iteritems():
        #        x_lsq_ma[i,coord[0],coord[1]] = x_lsq_a[j,i]
        tilt = x_lsq[0][(order+1)*x.size:].reshape((t.size, 3))
        f = plt.figure()
        ax = plt.gca()
        plt.title('Tilt correction magnitude')
        plt.axhline(0, color='k', linestyle=':')
        plt.axvline(0, color='k', linestyle=':')
        plt.scatter(tilt[:,0]*1E5, tilt[:,1]*1E5, c=tilt[:,2], s=16, cmap='RdBu', vmin=-2, vmax=2)
        plt.xlabel("X-tilt (m / 100 km)")
        plt.ylabel("Y-tilt (m / 100 km)")
        plt.colorbar(label="Z-offset (m)", extend='both')
        ax.set_aspect('equal')
        f_fn = os.path.splitext(stack_fn)[0]+'_tiltcorr_scatter.pdf'
        f.savefig(f_fn)
        ax.set_aspect('auto')
        plt.xlim(-2,2)
        plt.ylim(-0.1,0.1)
        f_fn = os.path.splitext(stack_fn)[0]+'_tiltcorr_scatter_zoom.pdf'
        f.savefig(f_fn)
        #3D plot of tilt
        #f = plt.figure()
        #ax = f.add_subplot(111)
        #ax = f.add_subplot(111, projection='3d')
        #ax.scatter(tilt[:,0], tilt[:,1], tilt[:,2])
    else:    
        #Extract coefficients
        x_lsq_a = x_lsq[0].reshape((x.size, order+1))
        #Populate original grids
        x_lsq_ma = np.ma.masked_all((order+1, test.shape[1], test.shape[2]))
        for i in range(order+1):
            x_lsq_ma[i,y,x] = x_lsq_a[:,i]

    #malib.iv(x_lsq_ma[1]*365.25, clim=(-3,3), cmap='RdBu')
    
    #Add the z offset back to the computed offset of linear dh/dt
    x_lsq_ma[0] += test_ref
    
    #malib.iv(x_lsq_ma[0])
    #malib.iv(x_lsq_ma[1]*365.25)
    #f = malib.iv(x_lsq_ma[1]*365.25, clim=(-3, 1))

    """
    #These are 
    #tilt_idx = np.array([10, 2, -2])
    tilt_idx = np.array([3, 2, -2])
    for i in tilt_idx:
        print tilt[i]
        print test[i]
        print apply_tilt(test[i], *tilt[i])
        print sample_lsq(x_lsq_ma, t[i] - t_ref)
    """
  
    #Some of these are nan
    xrefa = np.ma.fix_invalid(xrefa)
    yrefa = np.ma.fix_invalid(yrefa)
    xrefa_filled = xrefa.filled(0)
    yrefa_filled = yrefa.filled(0)
    tilt_idx = np.all(tilt == 0, axis=1)

    #Convert to original map coordinates
    #Add offset back to relative tilt
    tilt[:,2] -= tilt[:,0]*xrefa_filled + tilt[:,1]*yrefa_filled
        
    #Update stack source with 'tiltcorr'
    test_tiltcorr = np.ma.masked_all_like(test)
    test_tiltcorr_source = np.array(test_source, dtype='|S32')
    test_tilt = np.ma.masked_all_like(test)

    #Apply to the sampled ma
    #for i in range(test.shape[0]):
        ##test_tiltcorr[i] = apply_tilt(test[i], *tilt[i])
        #test_tiltcorr[i], test_tilt[i] = apply_tilt_map(test[i], gt, *tilt[i])

    #Make sure stride is 1!
    #if stride == 1:
    if True:
        print "Creating copies of input stack"
        stack_tiltcorr = deepcopy(stack)
        stack_tiltcorr_fn = os.path.splitext(stack_fn)[0] + '_lsq_tiltcorr.npz'
        stack_tiltcorr.stack_fn = stack_tiltcorr_fn 

        stack_tilt = deepcopy(stack)
        stack_tilt_fn = os.path.splitext(stack_fn)[0] + '_lsq_tilt.npz'
        stack_tilt.stack_fn = stack_tilt_fn 

        print "Applying correction"
        #validt_idx are the times that made it into test array
        for i,n in enumerate(validt_idx):
            out_tiltcorr, out_tilt = apply_tilt_map(stack.ma_stack[n], stack.gt, *tilt[i])
            stack_tiltcorr.ma_stack[n] = out_tiltcorr
            stack_tiltcorr.source[n] = stack_tiltcorr.source[n]+'_tiltcorr'
            stack_tilt.ma_stack[n] = out_tilt 
            stack_tilt.source[n] = stack_tilt.source[n]+'_tiltcorr'

        print "Writing out corrected stacks"
        stack_tiltcorr.compute_stats()
        stack_tiltcorr.write_stats()
        stack_tiltcorr.compute_trend()
        stack_tiltcorr.write_trend()
        stack_tiltcorr.savestack()

        print "Writing out final corrections to csv"
        out_fn = os.path.splitext(stack_tiltcorr_fn)[0] + '.csv'
        #out = tilt
        #hdr = 'tiltx,tilty,tiltz'
        out = zip(np.array(stack.fn_list)[validt_idx], stack.date_list[validt_idx], stack.date_list_o[validt_idx], \
                tilt[:,0], tilt[:,1], tilt[:,2]) 
        hdr = 'fn,date,date_o,tiltx,tilty,tiltz'
        np.savetxt(out_fn, out, delimiter=',', fmt='%s', header=hdr)
        #Should add filename and date fields

        print "Writing out tilt stack"
        stack_tilt.compute_stats()
        #stack_tilt.write_stats()
        stack_tilt.savestack()

        f = malib.iv(stack_tilt.stack_min, clim=(-5,5), cmap='RdBu', label='Tilt Correction Min (m)')
        f_fn = os.path.splitext(stack_fn)[0]+'_tilt_min.png'
        f.savefig(f_fn, dpi=300)
        f = malib.iv(stack_tilt.stack_max, clim=(-5,5), cmap='RdBu', label='Tilt Correction Max (m)')
        f_fn = os.path.splitext(stack_fn)[0]+'_tilt_min.png'
        f.savefig(f_fn, dpi=300)
        f = malib.iv(stack_tilt.stack_std, clim=(0,5), label='Std (m)')
        f_fn = os.path.splitext(stack_fn)[0]+'_tilt_std.png'
        f.savefig(f_fn, dpi=300)

        #Review tilts
        #for i in range(0,50):
        #    malib.iv(test_tilt[i], clim=(-2,2), cmap='RdBu')

        #Original stack stats
        f = malib.iv(stack.stack_trend, clim=(-3,1), label='Linear Trend (m/yr)')
        f_fn = os.path.splitext(stack_fn)[0]+'_tiltcorr_trend_before.png'
        f.savefig(f_fn, dpi=300)
        f = malib.iv(stack.stack_detrended_std, clim=(0,1), label='Detrended Std (m)')
        f_fn = os.path.splitext(stack_fn)[0]+'_tiltcorr_detrended_std_before.png'
        f.savefig(f_fn, dpi=300)
        #f = malib.iv(test.std(axis=0), clim=(0,5), label='Std (m)')
        f = malib.iv(stack.stack_std, clim=(0,5), label='Std (m)')
        f_fn = os.path.splitext(stack_fn)[0]+'_tiltcorr_std_before.png'
        f.savefig(f_fn, dpi=300)

        #Post-tiltcorr stats
        #test_tiltcorr_linreg = malib.ma_linreg(test_tiltcorr, np.ma.array(timelib.o2dt(t)))
        #f = malib.iv(test_tiltcorr_linreg[0], clim=(-3,1), label='Linear Trend (m/yr)')
        f = malib.iv(stack_tiltcorr.stack_trend, clim=(-3,1), label='Linear Trend (m/yr)')
        f_fn = os.path.splitext(stack_fn)[0]+'_tiltcorr_trend_after.png'
        f.savefig(f_fn, dpi=300)
        #f = malib.iv(test_tiltcorr_linreg[2], clim=(0,1), label='Detrended Std (m)')
        f = malib.iv(stack_tiltcorr.stack_detrended_std, clim=(0,1), label='Detrended Std (m)')
        f_fn = os.path.splitext(stack_fn)[0]+'_tiltcorr_detrended_std_after.png'
        f.savefig(f_fn, dpi=300)
        #f = malib.iv(test_tiltcorr.std(axis=0), clim=(0,5), label='Std (m)')
        f = malib.iv(stack_tiltcorr.stack_std, clim=(0,5), label='Std (m)')
        f_fn = os.path.splitext(stack_fn)[0]+'_tiltcorr_std_after.png'
        f.savefig(f_fn, dpi=300)

        #Computing tilt stats
        print "Computing tilt stats"
        r = stack_tilt.ma_stack
        r_flat = r.reshape(r.shape[0], r.shape[1]*r.shape[2])
        r_stats = np.vstack([r_flat.count(axis=1), r_flat.mean(axis=1), r_flat.std(axis=1), r_flat.ptp(axis=1)]).T

        #Split by source
        f = plt.figure()
        for lbl,m in [('DG_stereo', 's'), ('DG_stereo_nocorr', 'o'), ('DG_mono', 'D'), ('DG_mono_nocorr', 'd')]:
            idx = np.nonzero([bool(re.search(lbl+'$', i)) for i in test_source])[0]
            plt.errorbar(stack.date_list[idx], r_stats[idx,1], yerr=r_stats[idx,3]/2.0, linestyle='None', marker=m, label=lbl, markersize=5)
        plt.ylabel('DEM correction (m)')
        plt.title('DEM Offset and Tilt Magnitude')
        plt.axhline(0, color='k', linestyle=':')
        plt.legend(loc='lower right', prop={'size':10})
        ax = plt.gca()
        pltlib.fmt_date_ax(ax)
        pltlib.pad_xaxis(ax)
        plt.draw()
        f_fn = os.path.splitext(stack_fn)[0]+'_tiltcorr_offset+tilt.pdf'
        f.savefig(f_fn)

        #Should write this information out, sorting by biggest tilt/offset

    sys.exit()

    #malib.iv(np.ma.array(test.std(axis=0), mask=validmask), clim=(0,5))
    #malib.iv(np.ma.array(test.std(axis=0), mask=~validmask), clim=(0,5))

    #Should sort by size of tilt correction, then plot in ascending order, putting biggest on top

    z_tiltcorr = sample_lsq(x_lsq_ma, t - t_ref)
    zi = sample_lsq(x_lsq_ma, ti - t_ref)

    r = test_tiltcorr - z_tiltcorr
    r_flat = r.reshape(r.shape[0], r.shape[1]*r.shape[2])
    r_stats = np.vstack([r_flat.count(axis=1), r_flat.mean(axis=1), r_flat.std(axis=1)]).T
    r_stats = np.vstack([r_flat.count(axis=1), np.ma.median(r_flat, axis=1), np.ma.fix_invalid([malib.mad(i) for i in r_flat])]).T
    r_stats[r_stats[:,0] < 400] = np.ma.masked

    #malib.iv(r.std(axis=0))
    malib.iv(r.std(axis=0), clim=(0,1))
    rnmad = malib.mad_ax0(r)
    malib.iv(rnmad)

    if False:
        zi_origt = sample_lsq(x_lsq_ma, t - t_ref)
        zi_origt_r = test - zi_origt

        #Look at residuals for bad corrections
        zi_origt_r_2d = zi_origt_r.reshape((zi_origt_r.shape[0], zi_origt_r.shape[1]*zi_origt_r.shape[2]))
        zi_origt_r_med = np.ma.median(zi_origt_r_2d, axis=1)
        zi_origt_r_count = np.ma.count(zi_origt_r_2d, axis=1)
        zi_origt_r_nmad = [malib.mad(i) for i in zi_origt_r_2d]
        zi_origt_r_idx = np.arange(zi_origt_r_2d.shape[0])
        zi_origt_r_val = np.vstack([zi_origt_r_idx, zi_origt_r_count, zi_origt_r_med, zi_origt_r_nmad]).T
        #zi_origt_r_val[zi_origt_r_val[:,3].argsort()]

    contextfig, contextax = plt.subplots(1)
    contextax.imshow(x_lsq_ma[1]*365.25)
    random_idx = np.random.random_integers(0,y.shape[0]-1, 8)
    for i in random_idx:
        yi, xi = y[i],x[i]
        fig, ax = plt.subplots(1)
        title = 'Sample: (x=%s, y=%s)' % (xi, yi)
        #Note: stack slope/intercept are relative to 0 ordinal in years
        zi_orig = np.ma.array([test_linreg[0]*(i/365.25) + test_linreg[1] for i in ti])
        ax.set_title(title)
        ax.plot_date(ti, zi[:,yi,xi], color='b', marker=None, linestyle='--', label='LSQ dh/dt')
        ax.plot_date(ti, zi_orig[:,yi,xi], color='r', marker=None, linestyle='--', label='Orig dh/dt')
        ax.plot_date(t, test[:,yi,xi], color='r', markersize=5, label='Orig')
        ax.plot_date(t, test_tiltcorr[:,yi,xi], color='b', markersize=4, label='Corr')
        ax.legend(loc='lower center')
        pltlib.fmt_date_ax(ax)
        contextax.scatter(xi, yi, color='k')

sys.exit()

#Populate coordinate arrays for each timestep
xi = np.tile(x, ti.size)
yi = np.tile(y, ti.size)
ptsi = np.array((xi, yi, ti.repeat(x.size))).T

xin = rangenorm(xi, X.min(), X.ptp())
yin = rangenorm(yi, Y.min(), Y.ptp())
tin = rangenorm(ti, T.min(), T.ptp())

"""
#Radial basis function interpolation
#Need to normalize to input cube  
print "Running Rbf interpolation for %i points" % X.size
rbfi = scipy.interpolate.Rbf(Xn,Yn,Tn,VM, function='linear', smooth=0.1)
#rbfi = scipy.interpolate.Rbf(Xn,Yn,Tn,VM, function='gaussian', smooth=0.000001)
#rbfi = scipy.interpolate.Rbf(Xn,Yn,Tn,VM, function='inverse', smooth=0.00001)
print "Sampling result at %i points" % xin.size
vmi_rbf = rbfi(xin, yin, tin.repeat(x.size))
vmi_rbf_ma[:,y,x] = np.ma.fix_invalid(vmi_rbf.reshape((ti.size, x.shape[0])))
"""

#Attempt to load interpolation function 
#int_fn = 'LinearNDint_%i_%i_%i.pck' % (ti.size, test.shape[1], test.shape[2])
#Should add stack_fn here
#int_fn = 'LinearNDint_%s_%i_%i.pck' % (vtype, test.shape[1], test.shape[2]) 
int_fn = '%s_LinearNDint_%i_%i.pck' % (os.path.splitext(stack_fn)[0], test.shape[1], test.shape[2]) 
print int_fn
if load_existing and os.path.exists(int_fn):
    print "Loading pickled interpolation function: %s" % int_fn
    f = open(int_fn, 'rb')
    linNDint = pickle.load(f)
else:
    #LinearND interpolation
    print "Running LinearND interpolation for %i points" % X.size
    #Note: this breaks qhull for lots of input points - works for 500m and 1km grids over PIG shelf, not 256m
    linNDint = scipy.interpolate.LinearNDInterpolator(pts, VM, rescale=True)
    print "Saving pickled interpolation function: %s" % int_fn
    f = open(int_fn, 'wb')
    pickle.dump(linNDint, f, protocol=2)
    f.close()

"""
#NearestND interpolation (fast)
print "Running NearestND interpolation for %i points" % X.size
NearNDint = scipy.interpolate.NearestNDInterpolator(pts, VM, rescale=True)
"""

#vmi_fn = 'LinearNDint_%s_%i_%i_%iday.npy' % (vtype, test.shape[1], test.shape[2], ti_dt)
vmi_fn = '%s_%iday.npy' % (os.path.splitext(int_fn)[0], ti_dt)
if load_existing and os.path.exists(vmi_fn):
    print 'Loading existing interpolated stack: %s' % vmi_fn
    vmi_ma = np.ma.fix_invalid(np.load(vmi_fn))
else:
    #Once tesselation is complete, sample each timestep
    #Use multiprocessing here?
    #http://stackoverflow.com/questions/18597435/why-does-scipy-interpolate-griddata-hang-when-used-with-multiprocessing-and-open

    print "Sampling %i points at %i timesteps" % (x.size, ti.size)
    #Prepare array to hold output
    vmi_ma = np.ma.masked_all((ti.size, test.shape[1], test.shape[2]))

    """
    #This does all points at once
    #vmi = linNDint(ptsi)
    #vmi_ma[:,y,x] = np.ma.fix_invalid(vmi.reshape((ti.size, x.shape[0])))
    
    #This does interpolation serially by timestep
    for n, i in enumerate(ti):
        print n, i, timelib.o2dt(i)
        vmi_ma[n,y,x] = linNDint(x, y, i.repeat(x.size)).T
    """

    #Parallel
    pool = mp.Pool(processes=None)
    results = [pool.apply_async(dto_interp, args=(linNDint, x, y, i)) for i in ti]
    results = [p.get() for p in results]
    results.sort()
    for n, r in enumerate(results):
        print n, r[0], timelib.o2dt(r[0])
        vmi_ma[n,y,x] = r[1]

    vmi_ma = np.ma.fix_invalid(vmi_ma)
    print 'Saving interpolated stack: %s' % vmi_fn
    np.save(vmi_fn, vmi_ma.filled(np.nan))

origt = False 
if origt:
    print "Sampling %i points at %i original timesteps" % (x.size, t.size)
    vmi_ma_origt = np.ma.masked_all((t.size, test.shape[1], test.shape[2]))
    #Parallel
    pool = mp.Pool(processes=None)
    results = [pool.apply_async(dto_interp, args=(linNDint, x, y, i)) for i in t]
    results = [p.get() for p in results]
    results.sort()
    for n, r in enumerate(results):
        print n, r[0], timelib.o2dt(r[0])
        vmi_ma_origt[n,y,x] = r[1]

    vmi_ma_origt = np.ma.fix_invalid(vmi_ma_origt)
    #print 'Saving interpolated stack: %s' % vmi_fn
    #np.save(vmi_fn, vmi_ma.filled(np.nan))

#vmi = scipy.interpolate.griddata(pts, VM, ptsi, method='linear', rescale=True)

"""
#Kriging
#Should explore this more - likely the best option
#http://connor-johnson.com/2014/03/20/simple-kriging-in-python/
#http://resources.esri.com/help/9.3/arcgisengine/java/gp_toolref/geoprocessing_with_3d_analyst/using_kriging_in_3d_analyst.htm

#PyKrige does moving window Kriging, but only in 2D
#https://github.com/bsmurphy/PyKrige/pull/5

#Could do tiled kriging with overlap in parallel
#Split along x and y direction, preserve all t
#Need to generate semivariogram globally though, then pass to each tile
#See malib sliding_window
wx = wy = 30
wz = test.shape[0]
overlap = 0.5
dwx = dwy = int(overlap*wx)
gp_slices = malib.nanfill(test, malib.sliding_window, ws=(wz,wy,wx), ss=(0,dwy,dwx))

vmi_gp_ma = np.ma.masked_all((ti.size, test.shape[1], test.shape[2]))
vmi_gp_mse_ma = np.ma.masked_all((ti.size, test.shape[1], test.shape[2]))

out = []
for i in gp_slices:
    y, x = (i.count(axis=0) > 0).nonzero()
    x = x.astype(int)
    y = y.astype(int)
    vm_t = test[:,y,x]
    vm_t_flat = vm_t.ravel()
    idx = ~np.ma.getmaskarray(vm_t_flat)
    #These are values
    VM = vm_t_flat[idx]

    #These are coords
    X = np.tile(x, t.size)[idx]
    Y = np.tile(y, t.size)[idx]
    T = np.repeat(t, x.size)[idx]
    pts = np.vstack((X,Y,T)).T

    xi = np.tile(x, ti.size)
    yi = np.tile(y, ti.size)
    ptsi = np.array((xi, yi, ti.repeat(x.size))).T

    #gp = GaussianProcess(regr='linear', verbose=True, normalize=True, theta0=0.1, nugget=2)
    gp = GaussianProcess(regr='linear', verbose=True, normalize=True, nugget=2)
    gp.fit(pts, VM)
    vmi_gp, vmi_gp_mse = gp.predict(ptsi, eval_MSE=True)
    vmi_gp_ma = np.ma.masked_all((ti.size, i.shape[1], i.shape[2]))
    vmi_gp_ma[:,y,x] = np.array(vmi_gp.reshape((ti.size, x.shape[0])))
    vmi_gp_mse_ma = np.ma.masked_all((ti.size, i.shape[1], i.shape[2]))
    vmi_gp_mse_ma[:,y,x] = np.array(vmi_gp_mse.reshape((ti.size, x.shape[0])))
    out.append(vmi_gp_ma)
#Now combine intelligently

print "Gaussian Process regression"
pts2d_vm = vm_t[1]
pts2d = np.vstack((x,y))[~(np.ma.getmaskarray(pts2d_vm))].T
pts2di = np.vstack((x,y)).T
gp = GaussianProcess(regr='linear', verbose=True, normalize=True, theta0=0.1, nugget=1)
gp.fit(pts, VM)
print "Gaussian Process prediction"
vmi_gp, vmi_gp_mse = gp.predict(ptsi, eval_MSE=True)
print "Converting to stack"
vmi_gp_ma = np.ma.masked_all((ti.size, test.shape[1], test.shape[2]))
vmi_gp_ma[:,y,x] = np.array(vmi_gp.reshape((ti.size, x.shape[0])))
vmi_gp_mse_ma = np.ma.masked_all((ti.size, test.shape[1], test.shape[2]))
vmi_gp_mse_ma[:,y,x] = np.array(vmi_gp_mse.reshape((ti.size, x.shape[0])))
sigma = np.sqrt(vmi_gp_mse_ma)
"""

"""
#This fills nodata in last timestep with values from previous timestep
#Helps Savitzy-Golay filter
fill_idx = ~np.ma.getmaskarray(vmi_ma[-1]).nonzero()
temp = np.ma.array(vmi_ma[-2])
temp[fill_idx] = vmi_ma[-1][fill_idx]
vmi_ma[-1] = temp
"""

print "Running Savitzy-Golay filter"
sg_window = 5 
vmi_sg = malib.nanfill(vmi_ma, scipy.signal.savgol_filter, window_length=sg_window, polyorder=2, axis=0, mode='nearest')
#vmi_sg = malib.nanfill(vmi_ma, scipy.signal.savgol_filter, window_length=sg_window, polyorder=2, axis=0, mode='interp')

#[malib.iv(i, clim=(0,4000)) for i in vmi_ma]
#[malib.iv(vmi_ma[i] - vmi_sg[i], clim=(-100,100), cmap='RdBu') for i in range(vmi_ma.shape[0])]

#Now apply original mask
#if clip_to_shelfmask:
#    apply_mask(vmi_ma, m_orig)

#Now sample at hi-res
from scipy.ndimage.interpolation import map_coordinates
#Updated dt interval
ti_hi_dt = 20 
print "Interpolating with timestep of %0.2f days" % ti_hi_dt
#Interpolate at these points
ti_hi = np.arange(ti.min(), ti.max(), ti_hi_dt)
#This scales to appropriate indices for time axis of vmi_ma
ti_hi_idx = (ti.size-1)*(ti_hi - ti.min())/ti.ptp()
xi_hi = np.tile(x, ti_hi.size)
yi_hi = np.tile(y, ti_hi.size)
ptsi_hi = np.array((ti_hi_idx.repeat(x.size), yi_hi, xi_hi))
#Interpolate from sparse, regularly-gridded stack
#out_hi = map_coordinates(vmi_ma[:-2], ptsi_hi, mode='nearest', order=2) 
#out_hi = map_coordinates(vmi_ma, ptsi_hi, mode='nearest', order=1) 
out_hi = map_coordinates(vmi_sg, ptsi_hi, mode='nearest', order=1) 
#out_hi = malib.nanfill(vmi_ma, map_coordinates, ptsi_hi, mode='nearest', order=3) 
#out_hi = map_coordinates(vmi_ma.filled(vmi_ma.mean()), ptsi_hi, mode='nearest', order=2)
#Output array
out_ma = np.ma.masked_all((ti_hi.size, vmi_ma.shape[1], vmi_ma.shape[2]))
out_ma[:,y,x] = np.ma.masked_equal(out_hi.reshape((ti_hi.size, x.shape[0])), vmi_ma.fill_value)

#if clip_to_shelfmask:
#    apply_mask(out_ma, m_orig)

#print "Running Savitzy-Golay filter"
#sg_window = 9 
#vmi_sg = malib.nanfill(out_ma, scipy.signal.savgol_filter, window_length=sg_window, polyorder=2, axis=0, mode='nearest')

#This pulls out some random samples for evaluation
#Should also look at some profiles
contextfig, contextax = plt.subplots(1)
contextax.imshow(vmi_ma.mean(axis=0))
random_idx = np.random.random_integers(0,y.shape[0]-1, 10)
for i in random_idx:
    yi, xi = y[i],x[i]
    fig, ax = plt.subplots(1)
    title = 'Sample: (%s, %s)' % (xi, yi)
    plt.title(title)
    plt.plot_date(ti, vmi_sg[:,yi,xi], color='g', linestyle='-', label='LinearND_SG')
    #plt.plot_date(ti, vmi_gp_ma[:,yi,xi], color='m', linestyle='-', label='GPR')
    #plt.plot_date(ti, vmi_rbf_ma[:,yi,xi], color='m', linestyle='-', label='Rbf')
    #plt.plot_date(ti_hi, vmi_sg[:,yi,xi], color='g', linestyle='-', label='LinearND_SG')
    plt.plot_date(ti_hi, out_ma[:,yi,xi], color='k', linestyle='-', label='map_coord')
    plt.plot_date(ti, vmi_ma[:,yi,xi], color='b', linestyle='--', label='LinearND')
    plt.plot_date(t, test[:,yi,xi], color='r', linestyle=':', label='Orig')
    plt.legend(loc='lower right')
    pltlib.fmt_date_ax(ax)
    contextax.scatter(xi, yi, color='k')

#Apply original mask?

#Tensor product interpolation
#Series of 1-D interp, can use 'cubic'
#http://stackoverflow.com/questions/12618971/scipy-interpolate-linearndinterpolator-hangs-indefinitely-on-large-data-sets
#Then maybe savitzy-golay?

#http://gis.stackexchange.com/questions/173721/reconstructing-modis-time-series-applying-savitzky-golay-filter-with-python-nump
#http://www.sciencedirect.com/science/article/pii/S003442570400080X
#http://stackoverflow.com/questions/14119892/python-4d-linear-interpolation-on-a-rectangular-grid

#check out this one - regular gridding before map_coord
#http://stackoverflow.com/questions/16217995/fast-interpolation-of-regularly-sampled-3d-data-with-different-intervals-in-x-y/16221098#16221098

#https://github.com/JohannesBuchner/regulargrid
#http://stackoverflow.com/questions/6238250/multivariate-spline-interpolation-in-python-scipy?lq=1
#http://stackoverflow.com/questions/12618971/scipy-interpolate-linearndinterpolator-hangs-indefinitely-on-large-data-sets
