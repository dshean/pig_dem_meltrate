#! /usr/bin/env python

import sys
import os
import glob
import copy

from datetime import datetime, timedelta

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import stack_interp

from lib import iolib
from lib import malib
from lib import glaclib
from lib import timelib
from lib import geolib
from lib import filtlib

#Should look into Trackpy for this
#Or at least leverage pandas for tables of indices
#Issues where particle doesn't leave a particular pixel in the alloted time
#Anomalously high vdiv

#Define the minimum area of the intersection to proceed with Dh/Dt calculation
min_int_area = 1E7

firnair = 12

#Set this to find lsq dh/dt solution
lsq = False 

#Set this to limit to single velocity field
#False will interpolate velocities for each timestep
static_v = False

#Set this to output products at initial DEM1 pixel locations
init_dhdt = True

#Set this to compute median/nmad instead of mean/std for output products
robust = False

#Set this to apply SMB correction
smbcorr = False 

mask_fn = '/scr/pig_dem_stack/pig_shelf_poly_shean_2011gl.shp'
#mask_fn = '/Volumes/insar3/dshean/pig/cresis_atm_analysis/pig_shelf_poly_shean_2011gl.shp'

#These already exist
#dem_stack_fn = '20021204_1949_atm_mean-adj_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans-adj_stack_281.npz'
#Test stack with 2 DEMs
#dem_stack_fn = '20080105_SPI_09-009_PIG_SPOT_DEM_V1_40m_hae_fltr-trans_source-DEM-adj_tide_removed_20100118_GES_11-029_PIG_SPOT_DEM_V1_40m_hae_fltr-trans_source-DEM-adj_tide_removed_stack_2.npz'
#dem_stack_fn = '20080105_1447_SPI_09-009_PIG_SPOT_DEM_V1_40m_hae_newfltr-trans_source-DEM_tide_removed-adj_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans_tilt_removed-adj_stack_137.npz'
#dem_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/shelf_clip/20071020_1438_glas_mean-adj_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans-adj_stack_189.npz'
#dem_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/shelf_clip/20071020_1438_glas_mean-adj_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans-adj_stack_189_tideremoved.npz'
#dem_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/shelf_clip/nospirit_test/20091020_1718_lvis_mean-adj_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans-adj_stack_173.npz'

dem_stack_fn = '/scr/pig_stack_20160307_tworows_highcount/full_extent/new_workflow/more_testing/shelf_clip/merge_clip_filt_20080101-20150601.npz'
dem_stack_fn = sys.argv[1]

#These are annual stacks
#dem_stack_fn = '/scr/pig_stack_20160307_tworows_highcount/full_extent/new_workflow/more_testing/20021128_2050_atm_256m-DEM_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans_stack_730_tide_ib_mdt_geoid_removed_nocorr_offset_-3.10m_filt_DG+LVIS+ATM_2009-2016_lsq_tiltcorr_filt_nocorrinv_GLAS+SPIRIT_merge_extract/mos_year/20080101_20070701-20080630_mos-tile-0_20150101_20140701-20150630_mos-tile-0_stack_8_clip.npz'

#This was GPS site DEM test at 32 m/px
#dem_stack_fn = '/scr/pig_stack_20160307_tworows_highcount/gps_2012-2014_extent/20021204_1929_atm_32m-DEM_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans_stack_89_tide_ib_mdt_geoid_removed_nocorr_offset_-3.10m_lsq_tiltcorr_filt_nocorrinv_filt_20080101-20150601.npz'
#dem_stack_fn = '/scr/pig_stack_20160307_tworows_highcount/gps_2012-2014_extent/20021204_1929_atm_32m-DEM_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans_stack_89_tide_ib_mdt_geoid_removed_nocorr_offset_-3.10m_lsq_tiltcorr_filt_nocorrinv_filt_gpstest.npz'

#This is high-res channels extent 32 m/px
#dem_stack_fn = '/scr/pig_stack_20160307_tworows_highcount/gl_paper_extent/ian_extent/highres_mainshelf_channels/20021128_2050_atm_32m-DEM_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans_stack_210_tide_ib_mdt_geoid_removed_nocorr_offset_-3.10m_lsq_tiltcorr_filt_nocorrinv_clip_filt_20080101-20150601.npz'
#This is 64 m version
#dem_stack_fn = '/scr/pig_stack_20160307_tworows_highcount/gl_paper_extent/ian_extent/highres_mainshelf_channels/res64m/20080105_1447_SPI_09-009_PIG_SPOT_DEM_V1_40m_hae_newfltr-trans_source-DEM_20141121_1508_1020010039903900_10200100366CC300-DEM_32m_trans_stack_99.npz'

dem_dir = os.path.split(os.path.abspath(dem_stack_fn))[0]
dem_stack = malib.DEMStack(stack_fn=dem_stack_fn, save=False, med=False, trend=False, stats=True)

#vx_stack_fn = '20071101_1409_84days_20070910_0956-20071203_0925_alos_mos_Track-Pig07_vx_20140613_20140829_tsx_mos_vx_stack_16.npz'
#vx_stack_fn = os.path.join(v_dir, vx_stack_fn)
#vx_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vx_20150530_1556_22days_20150525_0356-20150605_0356_tsx_mos_track-MayJun15_vx_stack_19_clip.npz'
#vx_stack = malib.DEMStack(stack_fn=vx_stack_fn, save=False, trend=True, med=True, stats=True)
#vx_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vx_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vx_stack_22_clip_LinearNDint_121_142_121day.npz'
vx_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_512m/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vx_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vx_stack_22_clip_LinearNDint_237_277_121day.npz'
vx_stack = malib.DEMStack(stack_fn=vx_stack_fn, save=False, trend=False, med=False, stats=False, datestack=False)

#vy_stack_fn = '20071101_1409_84days_20070910_0956-20071203_0925_alos_mos_Track-Pig07_vy_20140613_20140829_tsx_mos_vy_stack_16.npz'
#vy_stack_fn = os.path.join(v_dir, vy_stack_fn)
#vy_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vy_20150530_1556_22days_20150525_0356-20150605_0356_tsx_mos_track-MayJun15_vy_stack_19_clip.npz'
#vy_stack = malib.DEMStack(stack_fn=vy_stack_fn, save=False, trend=True, med=True, stats=True)
#vy_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_1km/shelf_clip/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vy_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vy_stack_22_clip_LinearNDint_121_142_121day.npz'
vy_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/vel_20160225_512m/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vy_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vy_stack_22_clip_LinearNDint_237_277_121day.npz'
vy_stack = malib.DEMStack(stack_fn=vy_stack_fn, save=False, trend=False, med=False, stats=False, datestack=False)

#vm_stack_dt_list = vx_stack.date_list
vm_stack = np.ma.sqrt(vx_stack.ma_stack**2 + vy_stack.ma_stack**2)

if smbcorr: 
    smb_fn = '/scr/RACMO2.3_June2015_update/FDM_zs_ANT27_36y_1979-2014_zs_20080102_002_FDM_zs_ANT27_36y_1979-2014_zs_20141230_364_stack_1278.npz'
    smb_stack = malib.DEMStack(stack_fn=smb_fn, save=False)

#Shelf mask
m = geolib.shp2array(mask_fn, res=dem_stack.res, extent=dem_stack.extent)
dem_stack.ma_stack[:,m] = np.ma.masked

#Timestep
#dt = timedelta(days=5)
#dt_val = 20
dt_val = 10
vmax = np.percentile(np.ma.median(vm_stack, axis=0), 98)
max_dt_val = int(dem_stack.res/(vmax/365.25)) 
if dt_val > max_dt_val:
    dt_val = max_dt_val

#This was forced for high-res runs, otherwise interval is 2-3 days at 32 m/px
#dt_val = 10

dt = timedelta(days=dt_val)
#This is the timestep in units of years (should be a small number)
#Could be thought of as yr/dt: Multiply the m/yr or px/yr velocities by this 
dt_yr = dt.total_seconds()/timedelta(days=365.25).total_seconds()

#Define min and max time difference for Dh/Dt calculation
#min_t_diff_val = 0.5
#max_t_diff_val = 1.5
min_t_diff_val = 1.5
max_t_diff_val = 2.5
min_t_diff = timedelta(days=365.25*min_t_diff_val)
max_t_diff = timedelta(days=365.25*max_t_diff_val)

#These are min and max of DEM stack, used to interpolate velocities
#Used 0.5-2.5 years
min_t = min(dem_stack.date_list)
max_t = max(dem_stack.date_list)

#This is time range for GPS zoom
#min_t = datetime(2012,1,11) - timedelta(days=500)
#max_t = datetime(2013,12,20) + timedelta(days=500)

min_n_dt = int(min_t_diff.total_seconds()/dt.total_seconds())
max_n_dt = int(np.ceil(max_t_diff.total_seconds()/dt.total_seconds()))
total_n_dt = int(np.ceil((max_t - min_t).total_seconds()/dt.total_seconds()))

if (min_t + max_n_dt * dt) > max_t: 
    max_n_dt = total_n_dt

#Should attempt to deal with long intervals here
#dt_range = min_t + np.arange(max_n_dt) * dt
dt_range = min_t + np.arange(total_n_dt) * dt

#This is the last possible dem1 for wich we could successfully compute Dh/Dt with min_t_diff
max_t_init = max_t - min_n_dt * dt

#These are indices for all DEMs that should be run
dem_to_run = ((dem_stack.date_list <= max_t_init) & (dem_stack.date_list >= min_t)).data

#out_dir = 'meltrate_%iday_%0.1f-%0.1fyr' % (dt_val, min_t_diff_val, max_t_diff_val)
#out_dir = 'meltrate_%iday_%0.1f-%0.1fyr_variable_velocity' % (dt_val, min_t_diff_val, max_t_diff_val)
#out_dir = 'meltrate_%iday_%0.1f-%0.1fyr_variable_velocity_nearest' % (dt_val, min_t_diff_val, max_t_diff_val)
out_dir = 'meltrate_%iday_%0.1f-%0.1fyr_variable_velocity_interp_lsq_v512m_newrho_initdhdt' % (dt_val, min_t_diff_val, max_t_diff_val)
out_dir = os.path.join(dem_dir, out_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print
print "Timestep: %s" % dt
print "Time offset range: %s to %s" % (min_t_diff, max_t_diff)
print "Number of timesteps: %i to %i" % (min_n_dt, max_n_dt)
print "DEM time range: %s to %s" % (min_t, max_t)
print "Total number of timesteps: %i" % total_n_dt
print "DEM1 time range: %s to %s" % (min_t, max_t_init)
print
print "Output dir: %s" % out_dir
print

print "Populating velocities"

#Use median velocities for time period (good for testing)
#Could limit to relevant time period, but will potentially lose full coverge
if static_v:
    #Prepare velocity grids
    v_stack_dt_shape = list(vx_stack.ma_stack.shape)
    v_stack_dt_shape[0] = total_n_dt 
    print "Generating empty stacks to store data from each timestep"
    vx_stack_dt = np.ma.masked_all(v_stack_dt_shape)
    vy_stack_dt = np.ma.masked_all(v_stack_dt_shape)
    #vx_stack_dt[:] = np.ma.median(vx_stack.ma_stack, axis=0)
    #vy_stack_dt[:] = np.ma.median(vy_stack.ma_stack, axis=0)
    #Smooth median velocities
    print "Filling with median values"
    vx_stack_dt[:] = filtlib.gauss_fltr_astropy(np.ma.median(vx_stack.ma_stack, axis=0), 7)
    vy_stack_dt[:] = filtlib.gauss_fltr_astropy(np.ma.median(vy_stack.ma_stack, axis=0), 7)
else:
    print "Interpolating velocities at %0.1f day intervals" % dt_val
    vx_stack_dt = stack_interp.map_coord_interp(dt_range, dem_stack, vx_stack, filt=True)
    vy_stack_dt = stack_interp.map_coord_interp(dt_range, dem_stack, vy_stack, filt=True)
    #These are old hacks
    """
    #Time-variable velocity interpolation
    #Interpolates vx and vy for each timestep
    v_pad = 1.5 * 365.25
    for n, dt in enumerate(dt_range):
        print dt
        #This grabs nearest
        #vx_stack_dt[n] = stack_interp.dt_nearest(dt, vx_stack)
        #vy_stack_dt[n] = stack_interp.dt_nearest(dt, vy_stack)
        #This interpolates from nearest, but some are limited in spatial extent
        #vx_stack_dt[n] = stack_interp.dt_interp_nearest(dt, vx_stack)
        #vy_stack_dt[n] = stack_interp.dt_interp_nearest(dt, vy_stack)
        #This computes median from rolling window
        #vx_stack_dt[n] = stack_interp.dt_nearest_pad(dt, vx_stack, pad=v_pad)
        #vy_stack_dt[n] = stack_interp.dt_nearest_pad(dt, vy_stack, pad=v_pad)
        #This embeds nearest in median from rolling window
        #vx_stack_dt[n] = stack_interp.dt_nearest_pad(dt, vx_stack, pad=v_pad, embed=True)
        #vy_stack_dt[n] = stack_interp.dt_nearest_pad(dt, vy_stack, pad=v_pad, embed=True)
        #This computes rolling trend, then interpolates
        #This seems like best option, used for Dec 2015 variable velocity test
        #vx_stack_dt[n] = stack_interp.dt_interp_pad(dt, vx_stack, pad=v_pad)
        #vy_stack_dt[n] = stack_interp.dt_interp_pad(dt, vy_stack, pad=v_pad)
        #Try rolling trend, embedding short term and long term
        vx_stack_dt[n] = stack_interp.dt_interp_pad2(dt, vx_stack, pad=v_pad)
        vy_stack_dt[n] = stack_interp.dt_interp_pad2(dt, vy_stack, pad=v_pad)
    """

#vm_stack_dt = np.sqrt(vx_stack_dt**2+vy_stack_dt**2)

#Note: these should already be same res as dem_stack
#Convert m/yr velocity values to pixels/yr
vx_stack_dt = vx_stack_dt / dem_stack.res
#Note, Ian's velocities are flipped in y-direction
vy_stack_dt = -vy_stack_dt / dem_stack.res

#No longer need these, free up some memory
vx_stack = None
vy_stack = None

#Scale velocities for Polar stereographic grid?
#All indices are in relative pixel coords.  Might not actually matter much?
#For a given time interval of ~2 years, only propagating ~8 km, 1% is 80 m, that's less than 256 m pixel size
#Even for focused area, input pixels are still going to be larger than reality
#Need to scale to reduce, or increase velocity
#X and Y scaling will be different, can't use existing routines
#vxm, vym = geolib.get_xy_grids(vx_stack.get_ds())
#vlon, vlat, dummy = geolib.sps2ll(vxm, vym)
#vlat_scale = geolib.scale_ps(vlat)

#Now compute velocity divergence at each timestep
#Might be faster to loop through each grid along axis 0, avoid computing gradient along axis 0 unnecessarily
#Note, b/c input is 3D, x component is index 2, y component index 1
#vdiv_stack_dt = np.gradient(vx_stack_dt)[1] + np.gradient(vy_stack_dt)[0]
print "Computing velocity divergence"
#vdiv_stack_dt = np.ma.masked_all(v_stack_dt_shape)
#for i in np.arange(vx_stack_dt.shape[0]):
#    vdiv_stack_dt[i] = np.gradient(vx_stack_dt[i])[1] + np.gradient(vy_stack_dt[i])[0]
#dimensions are (t,y,x)
#vdiv = dU/dx + dV/dy
vdiv_stack_dt = np.gradient(vx_stack_dt)[2] + np.gradient(vy_stack_dt)[1]

#Note: should filter these up front
#Mask clearly bogus values in the velocity divergence
if False:
    max_vdiv = 0.2
    min_vdiv = -0.2
    print "Filtering velocity divergence (%0.2f to %0.2f)" % (min_vdiv, max_vdiv)
    vdiv_idx = (vdiv_stack_dt > max_vdiv) | (vdiv_stack_dt < min_vdiv) 
    vdiv_stack_dt[vdiv_idx.data] = np.ma.masked

#Should also prepare zs grids
#These could be true dzs for each dt interval
#Then just sum at end to compute ms component

#Need to limit the following to make sure we arent assigning the last dt_range value to a bunch of later dem_stack dates

print "Computing valid date range"
#Want to round true DEM timestamps to nearest dt
dem_date_list_dt_idx = []
dem_date_list_dt = []
for d in dem_stack.date_list:
    closest_d = timelib.get_closest_dt_idx(d, dt_range)
    dem_date_list_dt_idx.append(closest_d)
    dem_date_list_dt.append(dt_range[closest_d])

#Clean up indices here - don't assume i=0 is dem_stack[0]

print "Looping through valid DEMs"   
#for i, dem in enumerate(dem_stack.ma_stack[:-1]):
for i, dem1 in enumerate(dem_stack.ma_stack[dem_to_run]):
    dem1_fn = np.array(dem_stack.fn_list)[dem_to_run][i]
    dem1_t = dem_stack.date_list[dem_to_run][i]
    dem1_closest_t_idx = timelib.get_closest_dt_idx(dem1_t, dt_range)
    dem1_closest_t = dt_range[dem1_closest_t_idx]
    #valid_dem2_idx = np.logical_and(dt_range > dem1_closest_t + min_t_diff, dt_range < dem1_closest_t + max_t_diff).data.nonzero()[0]
    valid_dem2_idx = np.logical_and(dem_stack.date_list > dem1_closest_t + min_t_diff, dem_stack.date_list < dem1_closest_t + max_t_diff).data.nonzero()[0]
    if not valid_dem2_idx.size:
        continue
    valid_dem2_t = dem_stack.date_list[valid_dem2_idx] 
    valid_dem2_closest_t_idx = [timelib.get_closest_dt_idx(ii, dt_range) for ii in valid_dem2_t]
    valid_dem2_closest_t = dt_range[valid_dem2_closest_t_idx]

    my_dt_range = max(valid_dem2_closest_t) - dem1_closest_t

    #This returns tuples of x, y indices for all unmasked values
    #Note, these are pixel centers
    y_init, x_init = np.nonzero(~(np.ma.getmaskarray(dem1)))
    x = np.ma.array(x_init)
    y = np.ma.array(y_init)
    #vx = vx_stack_dt[t_idx][y, x]
    #vy = vy_stack_dt[t_idx][y, x]
    #vdiv = vdiv_stack_dt[t_idx][y, x]
    #zs
    #firnair

    #Create lists to record positions at each timestep
    #This is shape of array needed to store all timesteps
    t_shape = max(valid_dem2_closest_t_idx)+1 - dem1_closest_t_idx
    x_shape = (t_shape,) + x.shape
    #Note: using masked arrays here won't work
    #This attempts to preserve a single mask for the entire array
    #Want unique mask for each step
    x_hist = np.ma.masked_all(x_shape)
    y_hist = np.ma.masked_all_like(x_hist)
    #vx_hist = np.ma.masked_all_like(x_hist) 
    #vy_hist = np.ma.masked_all_like(x_hist) 
    vdiv_hist = np.ma.masked_all_like(x_hist) 

    print
    print '%i of %i: %s %s %s' % (i+1, dem_to_run.nonzero()[0].size, dem1_fn, dem1_t, dem1_closest_t) 

    #This loop precomputes all indices and extracts velocity divergence
    #The max(valid_dem2_closest_t_idx)+1 ensures that the last DEM history is written out
    for j in np.arange(dem1_closest_t_idx, max(valid_dem2_closest_t_idx)+1): 
        #This is the index for hist arrays (limited to smallest time range)
        h_idx = j - dem1_closest_t_idx
        #Extract vpx values for all indices
        #This rounds to nearest cell center
        #When (x,y) = (0,0), which is the cell center, value is extracted at (0,0)
        #When (x,y) = (0.5, 0.5), which is the cell boundary, value is extracted from the next cell at (1,1)
        #j_idx = [(y+0.5).astype(int), (x+0.5).astype(int)]

        vx = vx_stack_dt[j]
        vy = vy_stack_dt[j]
        vdiv = vdiv_stack_dt[j]
        
        #x_hist.append(x)
        #y_hist.append(y)
        #vdiv_hist.append(vdiv)
        x_hist[h_idx] = x
        y_hist[h_idx] = y
        #Could also just record these, then sample vdiv at later stage
        #vdiv_hist[h_idx] = vdiv[j_idx] 
        #This is bilinear interp
        vdiv_t = malib.nanfill(vdiv, scipy.ndimage.map_coordinates, [y,x], order=1, mode='nearest')
        vdiv_hist[h_idx] = vdiv_t

        #These should be interpolated at each step, not integers, use map_coordinates
        #Multiply by dt_yr, which is yr/dt, to convert to px/dt
        #vx_t = vx[j_idx]
        #vy_t = vy[j_idx]

        #Since vx and vy are continuous at each timestep, can interp
        vx_t = malib.nanfill(vx, scipy.ndimage.map_coordinates, [y,x], order=1, mode='nearest')
        vy_t = malib.nanfill(vy, scipy.ndimage.map_coordinates, [y,x], order=1, mode='nearest')

        #Note: vx_t and vy_t will have some huge bogus values
        #Should filter, but these are also handled by the clip step

        dx = vx_t * dt_yr
        dy = vy_t * dt_yr

        x = (x + dx)
        y = (y + dy)
    
        #Hack to deal with pixels that move beyond edge of grid
        #Should mask these rather than clip to edge indices
        #x = np.clip(x, 0, dem1.shape[1] - 1)
        #y = np.clip(y, 0, dem1.shape[0] - 1)
        #This is a better approach, but need to apply to original indices as well
        x = np.ma.masked_outside(x, 0, dem1.shape[1]-1)
        y = np.ma.masked_outside(y, 0, dem1.shape[0]-1)
        xy_mask = np.ma.getmaskarray(x) | np.ma.getmaskarray(y)
        x[xy_mask] = np.ma.masked
        y[xy_mask] = np.ma.masked

        #x_hist[:,xy_mask] = np.ma.masked
        #y_hist[:,xy_mask] = np.ma.masked
        #vdiv_hist[:,xy_mask] = np.ma.masked

    #This loop considers all candidate DEM2 grids
    for k, dem2_orig_idx in enumerate(valid_dem2_idx):
        dem2 = dem_stack.ma_stack[dem2_orig_idx]
        dem2_fn = dem_stack.fn_list[dem2_orig_idx]
        dem2_t = dem_stack.date_list[dem2_orig_idx] 
        dem2_closest_t_idx = valid_dem2_closest_t_idx[k]
        h_idx = valid_dem2_closest_t_idx[k] - dem1_closest_t_idx
        dem2_closest_t = valid_dem2_closest_t[k]

        #Note 2/15/16: these can have different number of unmasked entries (holes in velocity maps?)
        init_idx = np.ma.array([y_hist[0], x_hist[0]])
        final_idx = np.ma.array([y_hist[h_idx], x_hist[h_idx]])
        #init_idx_int = init_idx.astype(int)
        #final_idx_int = final_idx.astype(int) 
        #final_idx = [(y_hist[dem2_closest_t_idx]+0.5).astype(int), (x_hist[dem2_closest_t_idx]+0.5).astype(int)]
        #final_idx = [(y_hist[dem2_closest_t_idx].compressed()+0.5).astype(int), (x_hist[dem2_closest_t_idx].compressed()+0.5).astype(int)]
        #init_idx_masked = [np.ma.array(init_idx[0], mask=final_idx[0].mask), np.ma.array(init_idx[1], mask=final_idx[1].mask)]

        #xy_mask = np.ma.getmaskarray(init_idx[0]) | np.ma.getmaskarray(final_idx[0])

        #Invert to pull out valid entries
        xy_mask = ~(np.ma.getmaskarray(final_idx[0]))
  
        #init_idx = [init_idx[0].compressed(), init_idx[1].compressed()]
        #final_idx = [final_idx[0].compressed(), final_idx[1].compressed()]
        #init_idx = [init_idx[0][xy_mask].data, init_idx[1][xy_mask].data]
        #final_idx = [final_idx[0][xy_mask].data, final_idx[1][xy_mask].data]
        init_idx = init_idx[:,xy_mask].data
        final_idx = final_idx[:,xy_mask].data
        init_idx_int = init_idx.astype(int)
        final_idx_int = final_idx.astype(int) 

        #Extract DEM1 values
        #dem_init = dem1[init_idx_int]
        dem_init = malib.nanfill(dem1, scipy.ndimage.map_coordinates, [init_idx[0],init_idx[1]], order=1, mode='nearest')
        #Extract corresponding DEM2 values
        #dem_final = dem2[final_idx_int]
        dem_final = malib.nanfill(dem2, scipy.ndimage.map_coordinates, [final_idx[0],final_idx[1]], order=1, mode='nearest')
        int_area = dem_final.count() * dem_stack.res * dem_stack.res

        if int_area > min_int_area:

            if smbcorr:
                #Extract SMB
                smb_dem1_closest_t_idx = timelib.get_closest_dt_idx(dem1_t, smb_stack.date_list)
                smb_dem1 = smb_stack.ma_stack[smb_dem1_closest_t_idx]
                smb_dem2_closest_t_idx = timelib.get_closest_dt_idx(dem2_t, smb_stack.date_list)
                smb_dem2 = smb_stack.ma_stack[smb_dem2_closest_t_idx]

                #This if positive if the surface went up due to SMB
                smb_diff_eul = smb_dem2 - smb_dem1
                #smb_diff_eul_samp = smb_diff_eul[init_idx_int]
                smb_diff_eul_samp = malib.nanfill(smb_diff_eul, scipy.ndimage.map_coordinates, [final_idx[0],final_idx[1]], order=1, mode='nearest')
                #smb_init = smb_dem1[init_idx_int]
                #smb_final = smb_dem2[final_idx_int]
                #smb_diff = smb_final - smb_init

                smb_diff_eul = np.ma.array(smb_diff_eul, mask=np.ma.getmaskarray(dem1))

            #Compute actual time difference
            dem2_t_diff = dem2_t - dem1_t 
            dem2_t_diff_yr = dem2_t_diff.total_seconds()/timedelta(days=365.25).total_seconds()
            #The center time for the pair, used for output
            center_date = timelib.center_date(dem1_t, dem2_t)
            
            print
            print "Computing DH/Dt for: %s - %s (%i days)" % (dem1_t, dem2_t, dem2_t_diff.days) 

            #Compute Dh/Dt
            dh = dem_final - dem_init

            if False:
                #Mask outliers
                #This was done for larger, noisier SPIRIT grids, but throws out good values for WV grids
                clim = (0.1, 98)
                clim_minmax = malib.calcperc(dh, clim)
                dh[dh<clim_minmax[0]] = np.ma.masked
                dh[dh>clim_minmax[1]] = np.ma.masked

            if True:
                #Mask low elevations, likely open water or sea ice
                dem_final_thresh = 10.0
                dh[(dem_final < dem_final_thresh).data] = np.ma.masked

            #Eulerian dh/dt of original grids
            dh_eul = dem2 - dem1
            #Lagrangian Dh/Dt, using original DEM1 as reference coordinates
            dh_lag = np.ma.masked_all_like(dem1)
            #Populate with observed differences
            dh_lag[init_idx_int.tolist()] = dh

            dh_eul_rate = dh_eul/dem2_t_diff_yr
            dh_lag_rate = dh_lag/dem2_t_diff_yr

            if init_dhdt:
                init_vdiv_path = vdiv_hist[0:h_idx, xy_mask]
                #init_vdiv_path = vdiv_hist[0:h_idx, xy_mask][:,~(np.ma.getmaskarray(dh))].data

                #The right way to do this is to compute mean hvdiv along each path, then assign value to initial path loc
                #This computes h*vdiv at each timestep assuming steady dh/dt and observed vdiv on path
                #Then compute mean of h*vdiv history
                init_dh_ta = np.repeat(np.arange(h_idx), dem_init.shape[0]).reshape(h_idx, dem_init.shape[0])
                init_dh = (dem_init - firnair) + ((dh/dem2_t_diff_yr) * dt_yr * init_dh_ta)
                init_hvdiv_path = init_vdiv_path * init_dh
                init_hvdiv_path_std = init_hvdiv_path.std(axis=0)
                init_hvdiv_path = init_hvdiv_path.mean(axis=0)
                #Don't technically need this any longer
                init_vdiv_path = init_vdiv_path.mean(axis=0)

                #This uses midpoint elevation from observed lag_dh and dem_init
                """
                init_vdiv_path_std = init_vdiv_path.std(axis=0)
                init_vdiv_path = init_vdiv_path.mean(axis=0)
                init_hvdiv_path = init_vdiv_path * (dem_init - firnair + dh/2.0)
                init_hvdiv_path_std = init_vdiv_path_std * (dem_init - firnair + dh/2.0)
                """

                #Now populate maps
                init_vdiv = np.ma.masked_all_like(dem1)
                init_hvdiv = np.ma.masked_all_like(dem1)
                init_hvdiv_std = np.ma.masked_all_like(dem1)

                init_vdiv[init_idx_int.tolist()] = init_vdiv_path
                init_hvdiv[init_idx_int.tolist()] = init_hvdiv_path
                init_hvdiv_std[init_idx_int.tolist()] = init_hvdiv_path_std
            
                #Use closest vdiv
                #init_vdiv = vdiv_stack_dt[dem1_closest_t_idx]
                #init_hvdiv = init_vdiv * dem1

                #This uses mean of vdiv grids for entire period and midpoint thickness
                #These should all be continuous, don't need ma
                #init_vdiv = np.ma.median(vdiv_stack_dt[dem1_closest_t_idx:dem2_closest_t_idx], axis=0)
                #init_vdiv = vdiv_stack_dt[dem1_closest_t_idx:dem2_closest_t_idx].mean(axis=0)
                #init_vdiv_std = vdiv_stack_dt[dem1_closest_t_idx:dem2_closest_t_idx].std(axis=0)
                #init_hvdiv = init_vdiv * (dem1 - firnair + dh_lag/2.0)
                #init_hvdiv_std = init_vdiv_std * (dem1 - firnair + dh_lag/2.0)

            #This is used for testing of lsqr dh/dt calculation
            #Assumes vdiv is constant for the entire time period
            if lsq:
                #There is a bug somewhere that leads to discrepancy in number of records for the following
                #init_hvdiv_c = init_hvdiv.compressed()
                #This adds an extra record, which appears bogus
                init_hvdiv_c = init_hvdiv_path[~(np.ma.getmaskarray(dh))].data.T

            dh_c = dh.compressed()
            #Need to create combined mask for dh and vdiv
            vdiv_hist_c = vdiv_hist[0:h_idx, xy_mask][:,~(np.ma.getmaskarray(dh))].data.T

            #Compute final paths
            #Could probably go to int16
            x_hist_int = (x_hist[0:h_idx, xy_mask]+0.5).astype(np.int32)[:, ~(np.ma.getmaskarray(dh))].data
            y_hist_int = (y_hist[0:h_idx, xy_mask]+0.5).astype(np.int32)[:, ~(np.ma.getmaskarray(dh))].data

            path = np.dstack((x_hist_int.T, y_hist_int.T))

            #Throw out any paths that remain in the same cell for the full time
            print "Original path count: ", path.shape[0]
            #This is kind of a hack - probably more elegant way to do this
            path_nonzero = np.sum(np.sum(np.absolute(np.diff(path, axis=1)), axis=1), axis=1).nonzero()[0]
            path = path[path_nonzero]
            dh_c = dh_c[path_nonzero]
            #vdiv_hist_c = vdiv_hist_c[path_nonzero]
            print "Updated path count: ", path.shape[0]

            if path.shape[0] == 0:
                continue

            if lsq:
                init_hvdiv_c = init_hvdiv_c[path_nonzero]

            #Determine unique coordinates from all particle paths
            path_u = path.reshape(path.shape[0]*path.shape[1], path.shape[2])
            b = np.ascontiguousarray(path_u).view(np.dtype((np.void, path_u.dtype.itemsize * path_u.shape[1])))
            dummy, gidx, gc = np.unique(b, return_index=True, return_counts=True)
            guidx = path_u[gidx] 

            #Throw out any cells with only one path
            #guidx_gt1 = (gc > 1)
            #guidx = guidx[guidx_gt1]
            #gc = gc[guidx_gt1]

            #Create map of particle count within each cell
            path_count = np.ma.masked_all_like(dem1)
            path_count[guidx[:,1], guidx[:,0]] = gc

            if lsq:
                #Set up matrices to solve for dh/dt at each valid cell
                #Number of unique cells to assign a dh/dt value
                #Columns
                #53743
                N = guidx.shape[0]
                #Number of paths crossing these cells
                #Rows
                #42462
                M = path.shape[0]
                #Number of time steps
                #path.shape[1]

                print "Creating %i by %i matrices for dh/dt solution (num_paths, valid_cells)" % (M, N)

                cell_idx = {}
                for n,i in enumerate(guidx):
                    cell_idx[tuple(i)] = n

                A = scipy.sparse.lil_matrix((M,N))
                B = np.zeros(M)
                #B = dh_c/dem2_t_diff_yr
                #B = dh_c
                #This is Dh/Dt + h*vdiv at each cell
                #Resulting values will be mb rate
                #B = dh_c + (static_vdiv_c * (dem_init_c - firnair))
                B = dh_c + init_hvdiv_c

            #Create dictionaries to store values for unique indices
            #This is much more memory efficient than populating arrays
            print "Creating dictionaries to store output values"
            d_mb = {}
            for ix,iy in guidx:
                d_mb[(ix,iy)] = []
            d_DhDt = copy.deepcopy(d_mb)
            d_hvdiv = copy.deepcopy(d_mb)

            print "Looping through each particle path"
            #Instead of loop here, maybe define function and use apply_along_axis
            for n,p in enumerate(path):
                #This is a better way to determine amount of time spent in each pixel
                b = np.ascontiguousarray(p).view(np.dtype((np.void, p.dtype.itemsize * p.shape[1])))
                #dummy, idx, inv, c = np.unique(b, return_index=True, return_inverse=True, return_counts=True)
                dummy, idx, c = np.unique(b, return_index=True, return_counts=True)
                uidx = p[idx] 

                #If parcel encounters data gap in velocity map, it will "get stuck"
                #Should probably interpolate to fill holes
                #If parcel was in the same pixel the entire time, don't use it 
                #Not a valid Dh/Dt, but eulerian dh/dt
                
                #Note: 2/15/16 - now check for this before identifying unique cells
                if c.size < 2:
                    print "Found path that remains in one cell for entire period"
                    continue
                    
                if lsq:
                    A_idx = [cell_idx[tuple(u)] for u in uidx]

                    #This will solve for dh/dt at each cell
                    #Does not include h*vdiv
                    #A[n, A_idx] = c*dt_yr
                    #B[n] = dh_c[n]

                """
                pmin = np.min(p, axis=0)
                pmax = np.max(p, axis=0)
                bins = (pmax+1 - pmin).tolist()
                if bins != [0,0]:
                    h = np.histogram2d(p[:,0], p[:,1], bins, [[pmin[0], pmax[0]+1], [pmin[1], pmax[1]+1]]) 
                hma = np.ma.masked_equal(h[0].T, 0)

                dh * hma/path.shape[1]
                #path_idx = (path.ptp(axis=1).sum(axis=1) > 0)
                """

                #Extract initial DEM elevation for this particle
                #should probably update h based observed Dh/Dt for average
                dem1_init = dem1[p[0][1], p[0][0]] - firnair

                #Grab observed Dh/Dt for path
                #These values should be m/yr dh/dt in each cell 
                #Divide by the n*dt, rather than dem2_t_diff_yr?
                mydh = (dh_c[n] / dem2_t_diff_yr) 

                #This is estimate of height along the path due to observed Dh/Dt
                #Needed for h*vdiv
                dem1_path = dem1_init + mydh*(dt_yr * np.arange(p.shape[0]))

                #Extract vdiv along the path
                #These values should already be 1/yr
                #myvdiv = vdiv_hist_c[n, idx] * c/float(p.shape[0])/dt_yr
                #Note: using idx here only includes one vdiv value for cells with count > 1
                #This seems right, as the apparent melt rate will be higher, but the velocity divergence should be the same
                #myvdiv = vdiv_hist_c[n,idx] 
                myvdiv = vdiv_hist_c[n] 

                #Compute h*vdiv for each location along path
                #Need to multiply by the dem1 h above
                #Should really average vdiv within each pixel, as could evolve over time
                #hvdiv = dem1_init * vdiv_hist_c[n, idx]
                #hvdiv = dem1_init * myvdiv 
                hvdiv = dem1_path * myvdiv 

                mb = mydh + hvdiv
                
                if smbcorr:
                    mysmb = smb_diff_eul[p[0][1], p[0][0]]
                    mb -= mysmb

                if lsq:
                    #This solves for mb rate, rather than dh/dt
                    A[n, A_idx] = c*dt_yr
                    #B is populated up front
                    #Note: hvdiv is full path
                    #Could use first value, but can also define up front
                    #Not sure if dt_yr is necessary here
                    #B[n] = dh_c[n] + hvdiv*dt_yr - ms
                    #B[n] = dh_c[n] + hvdiv
                
                #Now write out to dictionaries
                for nu, u in enumerate(p):
                    #This writes the same average DhDt fraction to all cells visited
                    d_DhDt[tuple(u)].append(mydh)
                    #This is only needed if velocity is changing over the course of dem2_t_diff_yr
                    d_hvdiv[tuple(u)].append(hvdiv[nu])
                    d_mb[tuple(u)].append(mb[nu])

            #Output products
            out_fn_base = '%s_dt_%iday_%s_%s' % (center_date.strftime('%Y%m%d_%H%M'), dem2_t_diff.days, dem1_t.strftime('%Y%m%d_%H%M'), dem2_t.strftime('%Y%m%d_%H%M'))
            out_fn_base = os.path.join(out_dir, out_fn_base)

            if lsq:
                print "Preparing regularization terms"
                factor = 10
                E = np.full(N, 1./factor)
                E = scipy.sparse.diags(E, 0)
                E_b = np.zeros(N)
                print "Combining matrices"
                A = scipy.sparse.vstack([A,E])
                B = np.hstack([B,E_b])

            if lsq:
                #Add spatial smoothness constraint
                #This is pretty inefficient for large systems
                print "Preparing Smoothness Constraint"
                SC = scipy.sparse.lil_matrix((N,N))
                SC_b = np.zeros(N)
                #This sets order of constraint, use 1 for the linear term, 0 for the intercept 
                i = 1
                for n,(key,value) in enumerate(cell_idx.iteritems()):
                    key_up = (key[0]-1,key[1])
                    key_down = (key[0]+1,key[1])
                    key_left = (key[0],key[1]-1)
                    key_right = (key[0],key[1]+1)
                    ud = False
                    if (key_up in cell_idx) and (key_down in cell_idx):
                        SC[n,cell_idx[key]] = 2
                        SC[n,cell_idx[key_up]] = -1 
                        SC[n,cell_idx[key_down]] = -1 
                        ud = True
                    if (key_left in cell_idx) and (key_right in cell_idx):
                        if ud:
                            SC[n,cell_idx[key]] = 4
                        else:
                            SC[n,cell_idx[key]] = 2
                        SC[n,cell_idx[key_left]] = -1 
                        SC[n,cell_idx[key_right]] = -1 

                print "Combining matrices"
                A = scipy.sparse.vstack([A,SC])
                B = np.hstack([B,SC_b])

            if lsq:
                print "Solving for dh/dt with LSQ"
                print A.shape
                #Solve for dh/dt
                A = A.tocsr()
                mb_lsq_out = np.ma.masked_all_like(dem1)
                if True:
                    AT = A.T
                    umfpack = True
                    #This is a hack to get around UMFpack memory failure
                    if A.shape[0] > 600000:
                        umfpack = False
                    mb_lsq = scipy.sparse.linalg.spsolve(AT*A, AT*B, use_umfpack=umfpack)
                    for key, value in cell_idx.iteritems():
                        mb_lsq_out[key[::-1]] = mb_lsq[value]
                    out_fn = out_fn_base+'_mb_lsq.tif'
                    iolib.writeGTiff(mb_lsq_out, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
                else:
                    #Iterative LSQ
                    for iter in range(1,11):
                        mb_lsq = scipy.sparse.linalg.lsqr(A, B, iter_lim=iter, show=True)
                        #dhdt = scipy.sparse.linalg.lsqr(A, B, iter_lim=iter)
                        #Should print or preserve some of the information returned by lsqr
                        for key, value in cell_idx.iteritems():
                            mb_lsq_out[key[::-1]] = mb_lsq[0][value]
                        out_fn = out_fn_base+'_mb_lsq_%02diter.tif' % iter
                        iolib.writeGTiff(mb_lsq_out, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)

            print "Computing DhDt stats for all valid cells"
            DhDt_med = np.ma.masked_all_like(dem1)
            DhDt_nmad = np.ma.masked_all_like(dem1)
            for key, value in d_DhDt.iteritems():
                if robust:
                    DhDt_med[key[::-1]] = np.median(value)
                    DhDt_nmad[key[::-1]] = malib.mad(value)
                else:
                    DhDt_med[key[::-1]] = np.mean(value)
                    DhDt_nmad[key[::-1]] = np.std(value)

            print "Computing h_vdiv stats for all valid cells"
            hvdiv_med = np.ma.masked_all_like(dem1)
            hvdiv_nmad = np.ma.masked_all_like(dem1)
            for key, value in d_hvdiv.iteritems():
                if robust:
                    hvdiv_med[key[::-1]] = np.median(value)
                    hvdiv_nmad[key[::-1]] = malib.mad(value)
                else:
                    hvdiv_med[key[::-1]] = np.mean(value)
                    hvdiv_nmad[key[::-1]] = np.std(value)
            
            print "Computing mb stats for all valid cells"
            mb_med = np.ma.masked_all_like(dem1)
            mb_nmad = np.ma.masked_all_like(dem1)
            for key, value in d_mb.iteritems():
                if robust:
                    mb_med[key[::-1]] = np.median(value)
                    mb_nmad[key[::-1]] = malib.mad(value)
                else:
                    mb_med[key[::-1]] = np.mean(value)
                    mb_nmad[key[::-1]] = np.std(value)

            #Need to deal with surface melt
            #mb = dh_lag + dem_corr*v_div_avg - ms 

            #This is computed for each individual pixel 
            mb_out = glaclib.freeboard_thickness(-mb_med, clip=False)
            #This is computed from the output median grids
            #mb_med_out = glaclib.freeboard_thickness(-(DhDt_med + hvdiv_med), clip=False)
           
            print "Writing out products"
            print out_fn_base
            print

            if lsq:
                mb_lsq_out = glaclib.freeboard_thickness(-mb_lsq_out, clip=False)
                out_fn = out_fn_base+'_meltrate_lsq.tif'
                iolib.writeGTiff(mb_lsq_out, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)

            out_fn = out_fn_base+'_meltrate.tif'
            iolib.writeGTiff(mb_out, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
            #out_fn = out_fn_base+'_meltrate_medgrid.tif'
            #iolib.writeGTiff(mb_med_out, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)

            out_fn = out_fn_base+'_mb.tif'
            iolib.writeGTiff(mb_med, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
            out_fn = out_fn_base+'_mb_nmad.tif'
            iolib.writeGTiff(mb_nmad, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
            
            #Write out path_count
            out_fn = out_fn_base+'_count.tif'
            iolib.writeGTiff(path_count, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
            
            #Write out Dh/Dt 
            out_fn = out_fn_base+'_DhDt.tif'
            iolib.writeGTiff(DhDt_med, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
            out_fn = out_fn_base+'_DhDt_nmad.tif'
            iolib.writeGTiff(DhDt_nmad, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
            
            #Write out hvdiv
            out_fn = out_fn_base+'_hvdiv.tif'
            iolib.writeGTiff(hvdiv_med, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
            out_fn = out_fn_base+'_hvdiv_nmad.tif'
            iolib.writeGTiff(hvdiv_nmad, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)

            #Write out dh grids 
            out_fn = out_fn_base+'_dh_eul.tif'
            iolib.writeGTiff(dh_eul, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
            out_fn = out_fn_base+'_dh_lag.tif'
            iolib.writeGTiff(dh_lag, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
            #Write out annual rates
            out_fn = out_fn_base+'_dh_eul_rate.tif'
            iolib.writeGTiff(dh_eul_rate, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
            out_fn = out_fn_base+'_dh_lag_rate.tif'
            iolib.writeGTiff(dh_lag_rate, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)

            #These are grids that put all Dh/Dt at initial DEM pixel location
            #Avoids "smearing" and captures small-scale along-flow spatial variability
            if init_dhdt:
                init_mb = dh_lag_rate + init_hvdiv
                init_meltrate = glaclib.freeboard_thickness(-init_mb, clip=False)
                out_fn = out_fn_base+'_meltrate_init.tif'
                iolib.writeGTiff(init_meltrate, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
                out_fn = out_fn_base+'_vdiv_init.tif'
                iolib.writeGTiff(init_vdiv, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
                out_fn = out_fn_base+'_hvdiv_init.tif'
                iolib.writeGTiff(init_hvdiv, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)
                out_fn = out_fn_base+'_hvdiv_init_std.tif'
                iolib.writeGTiff(init_hvdiv_std, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)

            if smbcorr:
                #Write out zs grids
                out_fn = out_fn_base+'_dh_zs.tif'
                iolib.writeGTiff(smb_diff_eul, out_fn, gt=dem_stack.gt, proj=dem_stack.proj, ndv=-9999)

#After running,
#Set yearly
#mos_month_year.sh
