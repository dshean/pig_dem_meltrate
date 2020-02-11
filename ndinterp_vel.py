#! /usr/bin/env python

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
from scipy.ndimage.interpolation import map_coordinates

from lib import iolib
from lib import malib
from lib import timelib 
from lib import geolib
from lib import pltlib

#Specify the input type
vtype = 'vy'
#This will attempt to load cached files on disk
load_existing = False 
#Shelf mask
clip_to_shelfmask = True 

#This does the interpolation for a particular time for all points defined by x and y coords
#Used for parallel interpolation
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
#vx_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vx_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vx_stack_22_clip.npz'
vx_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_512m/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vx_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vx_stack_22_clip.npz'

#vy_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160222_500m/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vy_20150530_1556_22days_20150525_0356-20150605_0356_tsx_mos_track-MayJun15_vy_stack_20_clip.npz'
#vy_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160222_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vy_20150530_1556_22days_20150525_0356-20150605_0356_tsx_mos_track-MayJun15_vy_stack_20_clip.npz'
#vy_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vy_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vy_stack_22_clip.npz'
vy_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_512m/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vy_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vy_stack_22_clip.npz'

#vm_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160222_500m/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vm_20150530_1556_22days_20150525_0356-20150605_0356_tsx_mos_track-MayJun15_vm_stack_20_clip.npz'
#vm_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160222_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vm_20150530_1556_22days_20150525_0356-20150605_0356_tsx_mos_track-MayJun15_vm_stack_20_clip.npz'
#vm_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vm_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vm_stack_22_clip.npz'
vm_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vm_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vm_stack_22_clip.npz'
#vm_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_512m/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vm_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vm_stack_22_clip.npz'
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

#stack_fn = sys.argv[1]

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
    it = int(np.ceil(4000./stack.res))
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

if False:
    stride = 4 
    test = test[:,::stride,::stride]
    test_ptp = test_ptp[::stride,::stride]
    res *= stride 
    print "Using a stride of %i (%0.1f m)" % (stride, res)
    gt[[1,5]] *= stride

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

#test_med = malib.nanfill(test, np.nanmedian, axis=0)
#x,y,dummy = malib.get_xyz(test_med)
y, x = (test.count(axis=0) > 1).nonzero()
x = x.astype(int)
y = y.astype(int)
#vm_t = test.reshape(test.shape[0], test.shape[1]*test.shape[2])
vm_t = test[:,y,x]
vm_t_flat = vm_t.ravel()
idx = ~np.ma.getmaskarray(vm_t_flat)
#These are values
VM = vm_t_flat[idx]

#Determine scaling factors for x and y coords
#Should be the same for both 
xy_scale = max(x.ptp(), y.ptp())
xy_offset = min(x.min(), y.min())

#This scales t to encourage interpolation along the time axis rather than spatial axis
t_factor = 10. 
t_scale = t.ptp()*t_factor
t_offset = t.min()

xn = rangenorm(x, xy_offset, xy_scale)
yn = rangenorm(y, xy_offset, xy_scale)
tn = rangenorm(t, t_offset, t_scale)

X = np.tile(xn, t.size)[idx]
Y = np.tile(yn, t.size)[idx]
T = np.repeat(tn, x.size)[idx]
#These are coords
pts = np.vstack((X,Y,T)).T

#Interpoalte at these times
#ti = np.arange(t.min(), t.max(), 90.0)
ti_min = timelib.dt2o(datetime(2008,1,1))
#ti_min = timelib.dt2o(datetime(2006,9,1))
#ti_min = timelib.dt2o(datetime(2012,1,1))
#ti_max = timelib.dt2o(datetime(2009,1,1))
#ti_max = timelib.dt2o(datetime(2014,9,1))
#ti_max = timelib.dt2o(datetime(2015,4,1))
ti_max = timelib.dt2o(datetime(2015,6,1))
#ti_dt = 120 
ti_dt = 121.75 
#ti_dt = 91.3125
#ti_dt = 365.25 
#ti_dt = 365.25/2.0

#Interpolate at these times 
ti = np.arange(ti_min, ti_max, ti_dt)
tin = rangenorm(ti, t_offset, t_scale)
#ti = t
#Annual - use for discharge analysis?
#ti = timelib.dt2o([datetime(2008,1,1), datetime(2009,1,1), datetime(2010,1,1), datetime(2011,1,1), datetime(2012,1,1), datetime(2013,1,1), datetime(2014,1,1), datetime(2015,1,1)])

#Populate coordinate arrays for each timestep
#xi = np.tile(xn, ti.size)
#yi = np.tile(yn, ti.size)
#ptsi = np.array((xi, yi, tin.repeat(x.size))).T

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
    #linNDint = scipy.interpolate.LinearNDInterpolator(pts, VM, rescale=True)
    linNDint = scipy.interpolate.LinearNDInterpolator(pts, VM, rescale=False)
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
    #vmi_ma = np.ma.fix_invalid(np.load(vmi_fn))
    vmi_ma = np.ma.fix_invalid(np.load(vmi_fn)['arr_0'])
else:
    #Once tesselation is complete, sample each timestep
    #Use multiprocessing here?
    #http://stackoverflow.com/questions/18597435/why-does-scipy-interpolate-griddata-hang-when-used-with-multiprocessing-and-open

    print "Sampling %i points at %i timesteps, %i total" % (x.size, ti.size, x.size*ti.size)
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
    #results = [pool.apply_async(dto_interp, args=(linNDint, x, y, i)) for i in ti]
    results = [pool.apply_async(dto_interp, args=(linNDint, xn, yn, i)) for i in tin]
    results = [p.get() for p in results]
    results.sort()
    for n, r in enumerate(results):
        t_rescale = r[0]*t_scale + t_offset
        #print n, r[0], timelib.o2dt(r[0])
        print n, t_rescale, timelib.o2dt(t_rescale)
        vmi_ma[n,y,x] = r[1]

    vmi_ma = np.ma.fix_invalid(vmi_ma)
    print 'Saving interpolated stack: %s' % vmi_fn
    #np.savez(vmi_fn, vmi_ma.filled(np.nan))
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

#Write out a proper stack, for use by stack_melt and flux gate mass budget
if True:
    out_stack = deepcopy(stack)
    out_stack.stats = False
    out_stack.trend = False
    out_stack.datestack = False
    out_stack.write_stats = False
    out_stack.write_trend = False
    out_stack.write_datestack = False
    out_stack.ma_stack = vmi_ma
    out_stack.stack_fn = os.path.splitext(vmi_fn)[0]+'.npz'
    out_stack.date_list_o = np.ma.array(ti)
    out_stack.date_list = np.ma.array(timelib.o2dt(ti))
    out_fn_list = [timelib.print_dt(i)+'_LinearNDint.tif' for i in out_stack.date_list]
    out_stack.fn_list = out_fn_list
    out_stack.error = np.zeros_like(out_stack.date_list_o)
    out_stack.source = np.repeat('LinearNDint', ti.size)
    out_stack.gt = gt
    out_stack.res = res
    out_stack.savestack()

sys.exit()

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

#Note: moved the code below to stack_interp.py

#Now can interpolate at arbitrary coordinates in time/space
#Extract x,y from DEM stack extent/res and valid pixels
#Use defined ti range: min max, 20-day or 10-day
#Use map_coordinates to quickly sample

#Interpolate between these times 
#ti_hi_min = ti.min()
#ti_hi_max = ti.max()
ti_hi_min = timelib.dt2o(datetime(2008,1,3))
ti_hi_max = timelib.dt2o(datetime(2015,4,1))
#Updated dt interval
ti_hi_dt = 20 
ti_hi = np.arange(ti_hi_min, ti_hi_max, ti_hi_dt)

#This scales to appropriate indices for time axis of vmi_ma
#ti is the set of interpolated steps in vmi_ma
ti_hi_idx = (ti.size-1)*(ti_hi - ti.min())/ti.ptp()

#Need to scale x and y coords as well
dem_stack = malib.DEMStack(stack_fn=dem_stack_fn)
#Mask DEM stack
dem_mask = geolib.shp2array(mask_fn, res=dem_stack.res, extent=dem_stack.extent)
dem_x, dem_y = geolib.get_xy_grids(dem_stack.get_ds())
#These are final coordinates
dem_x = dem_x[~dem_mask]
dem_y = dem_y[~dem_mask]
#These are the output indices in DEM stack
out_dem_x, out_dem_y = geolib.mapToPixel(dem_x, dem_y, dem_stack.gt)
out_dem_x = out_dem_x.astype(int) 
out_dem_y = out_dem_y.astype(int) 

#Convert to vmi_ma coords
x_hi, y_hi = geolib.mapToPixel(dem_x, dem_y, gt)

#Prepare coordinate arrays for map_coordinates
xi_hi = np.tile(x_hi, ti_hi.size)
yi_hi = np.tile(y_hi, ti_hi.size)
ptsi_hi = np.array((ti_hi_idx.repeat(x_hi.size), yi_hi, xi_hi))

print "Interpolating with timestep of %0.2f days" % ti_hi_dt
out_hi = map_coordinates(vmi_ma, ptsi_hi, mode='nearest', order=1) 
#out_hi = malib.nanfill(vmi_ma, map_coordinates, ptsi_hi, mode='nearest', order=3) 
#out_hi = map_coordinates(vmi_ma.filled(vmi_ma.mean()), ptsi_hi, mode='nearest', order=2)

#Output array
out_ma = np.ma.masked_all((ti_hi.size, dem_stack.ma_stack.shape[1], dem_stack.ma_stack.shape[2]))
out_ma[:,out_dem_y,out_dem_x] = np.ma.masked_equal(out_hi.reshape((ti_hi.size, x_hi.size)), vmi_ma.fill_value)

sys.exit()

#Gaussian in time and spatial dimension?

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
    #plt.plot_date(ti_hi, out_ma[:,yi,xi], color='k', linestyle='-', label='map_coord')
    plt.plot_date(ti, vmi_ma[:,yi,xi], color='b', linestyle='--', label='LinearND')
    plt.plot_date(t, test[:,yi,xi], color='r', linestyle=':', label='Orig')
    plt.legend(loc='lower right', prop={'size':8})
    pltlib.fmt_date_ax(ax)
    contextax.scatter(xi, yi, color='k')
