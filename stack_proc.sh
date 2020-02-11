#! /bin/bash

#Run make_pig_stack on Pleiades
#Set to 256 m/px
#make_pig_stack_20160307.sh

orig_stack_fn=$1

stack_fn=$orig_stack_fn
fn_list=${stack_fn%.*}_fn_list.txt

#Add robust stats to stack error?

#This pulls out tides for center of PIG shelf
#Takes fn_list - no need to load full stack (although could do so to extract center coords for other locaitons)
#Writes out fn_list_tides.csv, which is read by stack_tidecorr.py
echo; echo "Extracting tide"
wv_tide.py $fn_list

#Pull out record of Pmsl for PIG shelf, used for IBE correction
#Writes out fn_list_IB.csv, which is read by stack_tidecorr.py
echo; echo "Extracting IBE"
ecmwf_proc.py $fn_list

#Remove tide, IBE and geoid offset
#Check MDT: currently set for -1.1 m
#Writes out ${stack_fn%.*}_tide_ib_mdt_geoid_removed.npz
#Also writes correction _tide_ib_mdt.npz and geoidoffset.tif grid
echo; echo "Removing tide, IBE, geoid"
stack_tidecorr.py $stack_fn
stack_fn=${stack_fn%.*}_tide_ib_mdt_geoid_removed.npz

#Remove mean bias from nocorr DEMs
#Writes out stack_fn_nocorr_offset_X.XXm.npz
echo; echo "Removing bias from nocorr WV DEMs"
stack_nocorr_adjust.py $stack_fn
#Can have +/- number
stack_fn=$(ls ${stack_fn%.*}_nocorr_offset_*\.[0-9][0-9]m.npz)

#Bypass tiltcorr if correcitons are already available
#Should also run melt and massbudget calculations without tiltcorr for comparison - get the same answer, reduced variance?

tiltcorr=true

if $tiltcorr ; then
    echo; echo "Isolating DG/LVIS/ATM 2009-2016 for tiltcorr"
    #Filter to isolate DG_LVIS_ATM 2009-present for tiltcorr
    stack_filter.py $stack_fn pig_pre_tiltcorr
    stack_fn=${stack_fn%.*}_filt_DG+LVIS+ATM_2009-2016.npz
    #This is residual filename, merged back in later
    stack_fn_spirit_glas=${stack_fn%.*}inv.npz

    #Run LSQ tilt/offset correction
    #Check stride is 1 for final output
    #Check that shelf is used
    #Run with or without shelf

    #Note: still using malib.iv here, so figures must be closed to proceed
    echo; echo "Running tiltcorr"
    ndinterp.py $stack_fn
    stack_fn=${stack_fn%.*}_lsq_tiltcorr.npz
else
    #tiltcorr_fn=$(ls *tiltcorr.csv)
    #tiltcorr_fn=/scr/pig_stack_20160307_tworows_highcount/full_extent/DG_LVIS_ATM_only_fortiltcorr/20091020_1718_lvis_256m-DEM_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans_stack_355_nocorr_offset_-3.10m_lsq_tiltcorr.csv
    tiltcorr_fn=/scr/pig_stack_20160307_tworows_highcount/full_extent/new_workflow/more_testing/20021128_2050_atm_256m-DEM_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans_stack_730_tide_ib_mdt_geoid_removed_nocorr_offset_-3.10m_filt_DG+LVIS+ATM_2009-2016_lsq_tiltcorr.csv
    #tiltcorr_fn=${stack_fn%.*}_lsq_tiltcorr.csv
    echo; echo "Applying existing tiltcorr corrections"
    apply_lsq_tiltcorr.py $stack_fn $tiltcorr_fn
    stack_fn=${stack_fn%.*}_lsq_tiltcorr.npz
fi

#Remove any remaining nocorr, bad mono
stack_filter.py $stack_fn remove_nocorr
stack_fn=${stack_fn%.*}_nocorrinv.npz

if $tiltcorr ; then
    #Re-Merge with SPIRIT, GLAS
    out_fn=${stack_fn%.*}_GLAS+SPIRIT_merge.npz
    stack_merge.py $stack_fn $stack_fn_spirit_glas $out_fn
    stack_fn=$out_fn
fi

#Make annual stacks
#stack_annual.py $stack_fn
#Extract individual DEMs
stack_extract.py $stack_fn
cd ${stack_fn%.*}_extract
mkdir hs
mv *hs*tif hs/
mos_month_year.sh 
cd ..

clipdir=shelf_clip
if [! -d $clipdir] ; then 
    mkdir $clipdir
fi

#Clip to shelf
stack_clip.py $stack_fn
mv ${stack_fn%.*}_clip* $clipdir

#Limit to relevant dates
#Throw out early GLAS
#Could just filter GLAS out entirely, maybe mosaic after tidecorr?

#Velocity stack prep
#Set vtype to vx, vy
#Select stack resolution - 512 m is good compromise
#Clip to shelfmask = True, this will pad by 4 km
#This will spit out vx and vy stacks with 120 day interval
ndinterp_vel.py

#Melt rate calculation
#Update vx and vy stack fn
#Run with 0.5-1.5 yr
stack_melt_lsq_path.py $stack_fn
#Run with 1.5-2.5 yr
stack_melt_lsq_path.py $stack_fn

#Create annual DEM interpolation
ndinterp_dem.py

#Extract profiles at gates
fluxgate_prep.sh
