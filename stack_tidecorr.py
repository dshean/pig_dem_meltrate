#! /usr/bin/env python

import os
import sys

import numpy as np
import scipy.ndimage
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from lib import iolib
from lib import malib
from lib import geolib
from lib import filtlib
from lib import pltlib
from lib import timelib

#This finds candidate pairs for tide+IBE DEM difference
def find_pairs(dem_stack, tide_list, max_dt=timedelta(450.0)):
    ds = dem_stack.get_ds()
    line_fn = '/scr/pig_dem_stack/pig_shelf_poly_shean_2011gl_line.shp'
    m = geolib.shp2array(line_fn, res=dem_stack.res, extent=dem_stack.extent)
    #This is total gl pixels
    #max_gl_count = (~m).nonzero()[0].size
    pairs = []
    dt_list = dem_stack.date_list
    min_dz = 0.5 
    #This is 100 km2
    min_count = 2000
    min_gl_count = 40
    out = []
    for i,dt in enumerate(dt_list):
        candidate_idx = timelib.get_closest_dt_padded_idx(dt, dt_list, max_dt)
        tidediff = tide_list[i] - tide_list[candidate_idx]
        valid_candidate_idx = np.abs(tidediff) > min_dz
        if np.any(valid_candidate_idx):
            for c in candidate_idx[valid_candidate_idx]:
                dz = dem_stack.ma_stack[i] - dem_stack.ma_stack[c]
                if dz.count() > min_count and (dz * ~m).nonzero()[0].size > min_gl_count:
                    if not (c,i) in pairs:
                        #if dem_stack.source[i] != 'glas' and dem_stack.source[c] != 'glas' and dem_stack.source[i] != 'spirit' and dem_stack.source[c] != 'spirit':
                        #if 'DG_stereo' in dem_stack.source[i] and 'DG_stereo' in dem_stack.source[c]:
                        if 'DG' in dem_stack.source[i] and 'DG' in dem_stack.source[c]:
                            dt_diff = (dt_list[c]-dt_list[i]).total_seconds()/86400.
                            tide_diff = tide_list[i]-tide_list[c]
                            title = '%s-%s (%0.1f days) dz=%0.2f m' % (timelib.print_dt(dt_list[i]), timelib.print_dt(dt_list[c]), dt_diff, tide_diff) 
                            print title, i, c
                            pairs.append((i,c))
                            flip=1.0
                            if tide_diff < 0:
                                flip=-1.0
                            o = flip*(dz-np.ma.median(dz))
                            out.append(o)
                            f = malib.iv(o, clim=(-1.0, 1.0), cmap='RdYlBu', title=title)
                            pltlib.shp_overlay(f.gca(), ds, line_fn, color='green')
    out = np.ma.array(out)
    malib.iv(np.ma.median(out, axis=0), clim=(-1.0, 1.0), cmap='RdYlBu')
    return pairs, out

#mask_fn = '/Volumes/insar3/dshean/pig/cresis_atm_analysis/pig_shelf_poly_shean_2011gl.shp'
#mask_fn = '/scr/pig_dem_stack/pig_shelf_poly_shean_2011gl.shp'
#This extends out to edge of domain - should eliminate shelf edge effects
mask_fn = '/scr/pig_dem_stack/pig_shelf_poly_shean_2011gl_edge.shp'

#dem_stack_fn = '/scr/pig_stack_20151201_tworows_highcount/shelf_clip/20071020_1438_glas_mean-adj_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans-adj_stack_189.npz'
#dem_stack_fn = '/scr/pig_stack_20160307_tworows_highcount/shelf_clip/20021128_2050_atm_256m-DEM_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans_stack_417.npz' 
#dem_stack_fn = '/scr/pig_stack_20160307_tworows_highcount/full_extent/20021128_2050_atm_256m-DEM_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans_stack_730.npz'

dem_stack_fn = sys.argv[1]
dem_stack = malib.DEMStack(stack_fn=dem_stack_fn, save=False, med=False, trend=True, stats=True)

#Tide: this is expected displacement of surface due to tide 
#Want to subtract this to correct

#Run wv_tide.py to extract tide for center of shelf
#tide_csv_fn = '/scr/pig_stack_20151201_tworows_highcount/shelf_clip/20071020_1438_glas_mean-adj_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans-adj_stack_189_fn_list_tides.csv'
tide_csv_fn = os.path.join(os.path.splitext(dem_stack_fn)[0]+'_fn_list_tides.csv')
tide = np.genfromtxt(tide_csv_fn, dtype=None, delimiter=',', names=True)
tide_full_csv_fn = os.path.join(os.path.splitext(dem_stack_fn)[0]+'_fn_list_tides_hr.csv')
tide_full = np.genfromtxt(tide_full_csv_fn, dtype=None, delimiter=',', names=True)

#IB: this is expected displacement of surface due to pressure
#Want to subtract this to correct

#ib_csv_fn = '/Volumes/insar3/dshean/ECMWF/pig_ecmwf_IB_2002-2015.csv'
#ib_csv_fn = '/scr/pig_stack_20160307_tworows_highcount/shelf_clip/20021128_2050_atm_256m-DEM_20150223_1531_1040010008556800_104001000855E100-DEM_32m_trans_stack_417_fn_list_IB.csv'
ib_csv_fn = os.path.join(os.path.splitext(dem_stack_fn)[0]+'_fn_list_IB.csv')
ib = np.genfromtxt(ib_csv_fn, dtype=None, delimiter=',', names=True)
ib_full_csv_fn = '/Volumes/insar3/dshean/ECMWF/pig_ecmwf_IB_2002-2016.csv'
ib_full = np.genfromtxt(ib_full_csv_fn, dtype=None, delimiter=',', names=True)

#Create figure of corrections
if False:
    from scipy.interpolate import interp1d
    f = interp1d(ib_full['date_o'],ib_full['ecmwf_IB'], kind='linear', bounds_error=False)
    xi = tide_full['date_o'] 
    yi = f(xi)

    combined = yi + tide_full['tide']

    f, ax = plt.subplots()
    ax.plot_date(tide['date_o'],tide['tide'],label='Tide')
    ax.plot_date(ib['date_o'],ib['IB'],label='IB')
    ax.plot_date(tide['date_o'], tide['tide']+ib['IB'], label='Combined')
    ax.set_ylabel('Elevation (m)')
    ax.legend()
    plt.show()

    f, axa = plt.subplots(3,1,sharex=True,figsize=(10,7.5))
    out_fn = os.path.splitext(dem_stack_fn)[0]+'_tide_IB_fig.png'
    axa[0].plot_date(tide_full['date_o'], tide_full['tide'],marker=None,linestyle='-',linewidth=0.5,color='b',alpha=0.5)
    axa[0].plot_date(tide['date_o'], tide['tide'],marker='o',markersize=4,color='b')
    axa[0].set_ylabel('Tide (m)')
    axa[1].plot_date(ib_full['date_o'], ib_full['ecmwf_IB'],marker=None,linestyle='-',linewidth=0.5,color='g',alpha=0.5)
    axa[1].plot_date(ib['date_o'], ib['IB'],marker='o',markersize=4,color='g')
    axa[1].set_ylabel('Inv. Barometer (m)')
    axa[2].plot_date(tide_full['date_o'], combined,marker=None,linestyle='-',linewidth=0.5,color='r',alpha=0.5)
    axa[2].plot_date(ib['date_o'], tide['tide']+ib['IB'],marker='o',markersize=4,color='r')
    axa[2].set_ylabel('Combined (m)')
    axa[0].set_xlim(ib['date_o'].min(), ib['date_o'].max())
    pltlib.fmt_date_ax(axa[0])
    pltlib.fmt_date_ax(axa[1])
    pltlib.fmt_date_ax(axa[2])
    pltlib.pad_xaxis(axa[2])
    plt.savefig(out_fn, dpi=300)
    plt.show()

#pairs, out = find_pairs(dem_stack, tide['tide']+ib['IB'], max_dt=timedelta(180))

#Want to subtract this from WGS84 elevation
#mdt = -1.1
mdt = 0

geoid = 0
ds = dem_stack.get_ds()
geoid_fn = os.path.splitext(dem_stack_fn)[0]+'_geoidoffset.tif'
#Want to add this to WGS84 elevation
geoid = geolib.dem_geoid_offsetgrid_ds(ds, geoid_fn)

#Could create new stack of GL position
print "Creating shelf mask with feathered edge"
fwidth_m = 3000.
#Shelf mask
fwidth = int((fwidth_m/dem_stack.res)+0.5)
m = geolib.shp2array(mask_fn, res=dem_stack.res, extent=dem_stack.extent)
m_d = scipy.ndimage.binary_dilation(m, iterations=int(fwidth/2.)).astype(float)
#m_d_f = filtlib.gauss_fltr_astropy(m_d, size=7)
m_d_f = scipy.ndimage.filters.uniform_filter(m_d, size=fwidth)
m_d_f = 1.0 - m_d_f

print "Creating copy of DEM stack"
from copy import deepcopy
tide_stack = deepcopy(dem_stack)
#tide_stack.stack_fn = os.path.splitext(dem_stack.stack_fn)[0]+'_tide.npz'
tide_stack.stack_fn = os.path.splitext(dem_stack.stack_fn)[0]+'_tide_ib_mdt.npz'

print "Populating correction stack"
#tide_stack.ma_stack[:,m] = np.ma.masked
#Want to be careful here - make sure every record in dem_stack has tide and ib
for i in range(dem_stack.ma_stack.shape[0]):
    #ta = np.ma.array(np.full(dem_stack.ma_stack[0].shape, tide[i][-1]), mask=m_d, fill_value=0)
    #ta = np.full(dem_stack.ma_stack[0].shape, tide[i][-1])
    ta = np.full(dem_stack.ma_stack[0].shape, tide[i][-1] + ib[i][-1] + mdt)
    ta_smooth = ta * m_d_f
    tide_stack.ma_stack[i][:] = ta_smooth

print "Creating copy of DEM stack"
from copy import deepcopy
dem_stack_tidecorr = deepcopy(dem_stack)
dem_stack_tidecorr.stack_fn = os.path.splitext(dem_stack.stack_fn)[0]+'_tide_ib_mdt_geoid_removed.npz'
print "Applying correction"
dem_stack_tidecorr.ma_stack = dem_stack.ma_stack - tide_stack.ma_stack + geoid

for i in (dem_stack_tidecorr, tide_stack):
    i.compute_stats()
    i.write_stats()
    i.compute_trend()
    i.write_trend()
    i.savestack()

#pairs, out = find_pairs(dem_stack_tidecorr, tide['tide']+ib['IB'], max_dt=timedelta(90))
