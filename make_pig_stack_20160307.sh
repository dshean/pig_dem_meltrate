#! /bin/bash

fn=''

#These were coregistered to median WV DEM over region to the N of the shelf
#These are aligned, two separate ways, should find align correctly
#spirit_dir=/nobackupp8/deshean/SPIRIT/dem/reprocess_20151013/pig_spirit_20151201_combined
#spirit_fn=" $(ls $spirit_dir/*newfltr_trans-adj.tif)"
#Original alignment to ptdb
#Note: 20090103_1442_SPI_09-046_PIG_SPOT_DEM_V1_40m is bad
spirit_dir=/nobackupp8/deshean/SPIRIT/dem/reprocess_20151013
#spirit_fn=" $(ls $spirit_dir/*newfltr-trans_source-DEM-adj.tif)"
#spirit_fn=" $(ls $spirit_dir/*newfltr-trans_source*-DEM-adj.tif)"
#Throw out the upsream stuff - just adds noise
#20090220 SPIRIT DEM covers the area just upstream of the grounding line - maybe crop and remove other regions
#20110123 covers S side of trunk upstream of gl, but lots of WV coverage 
#spirit_fn=" $(ls $spirit_dir/*newfltr-trans_source*-DEM-adj.tif | grep -v 20090131 | grep -v 20110123 | grep -v 20090220 | grep -v 20080315)"

#spirit_fn=" $(ls $spirit_dir/*newfltr-trans_source*-DEM.tif | grep -v 20090131 | grep -v 20110123 | grep -v 20080315)"
spirit_fn=" $(ls $spirit_dir/*newfltr.tif | grep -v 20090131 | grep -v 20110123 | grep -v 20080315)"

echo $(echo $spirit_fn | wc -w) SPIRIT
echo $spirit_fn | tr ' ' '\n'
echo
fn+=" $spirit_fn"

#This is gridded altimetry resolution to use
ptres=256

#ptdb_dir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment/pig_ptdb_20151026_2153_extrarows/full_extrarows_extent_256m/
ptdb_dir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment/pig_ptdb_20151026_2153_extrarows/cluster_0.5day/
#Shoudl rerun with new extent at higher res
#ptdb_dir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment/pig_ptdb_20151026_2153_extrarows
#atm_fn=" $(ls $ptdb_dir/*atm_mean.tif)"
#atm_fn=" $(ls $ptdb_dir/*atm_mean-adj.tif)"
atm_fn=" $(ls $ptdb_dir/*atm_${ptres}m-DEM.tif)"
echo $(echo $atm_fn | wc -w) ATM 
echo $atm_fn | tr ' ' '\n'
echo
fn+=" $atm_fn"
#lvis_fn=" $(ls $ptdb_dir/*lvis_mean.tif)"
#lvis_fn=" $(ls $ptdb_dir/*lvis_mean-adj.tif)"
lvis_fn=" $(ls $ptdb_dir/*lvis_${ptres}m-DEM.tif)"
echo $(echo $lvis_fn | wc -w) LVIS 
echo $lvis_fn | tr ' ' '\n'
echo
fn+=" $lvis_fn"

#These are grouped by campaign
ptdb_dir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment/pig_ptdb_20151026_2153_extrarows/cluster_16day/
#These are grouped by individual orbit
#ptdb_dir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment/pig_ptdb_20151026_2153_extrarows/cluster_0.1day_glas/
#glas_fn=" $(ls $ptdb_dir/*glas_mean.tif)"
#glas_fn=" $(ls $ptdb_dir/*glas_mean-adj.tif)"
glas_fn=" $(ls $ptdb_dir/*glas_${ptres}m-DEM.tif)"
echo $(echo $glas_fn | wc -w) GLAS 
echo $glas_fn | tr ' ' '\n'
echo
fn+=" $glas_fn"

#NEED TO GEOID
#NEED TO TIDE
#NEED to TILTCORR
#NEED TO update non-trans

mono_dir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment_extended_rows/mono_ln
mono_fn=''
prefix_list=$(ls $mono_dir/*DEM_mono_32m.tif | awk -F'-' '{print $1}' | sort -u)
#prefix_list=$(ls $mono_dir/*DEM_mono_32m.tif | awk -F'-' '{print $1}' | sort -u | head -1)
for i in $prefix_list 
do
    #if [ -e ${i}-DEM_mono_32m_trans_tilt_removed-adj.tif ] ; then
    #    mono_fn+=" ${i}-DEM_mono_32m_trans_tilt_removed-adj.tif"
    #if [ -e ${i}-DEM_mono_32m_trans.tif ] ; then
    #    mono_fn+=" ${i}-DEM_mono_32m_trans.tif"
    #else
        mono_fn+=" ${i}-DEM_mono_32m.tif"
    #fi
done
echo $(echo $mono_fn | wc -w) WV mono 
echo $mono_fn | tr ' ' '\n'
echo
fn+=" $mono_fn"

#These are outside the central two rows
stereo_list_tocut=/nobackupp8/deshean/wais_coastline/dem/pig_catchment_extended_rows/wais_coastline_all_32m_pig_catchment_extended_rows_20151030_tocut_name.txt
stereo_list_tocut_lowcount=/nobackupp8/deshean/wais_coastline/dem/pig_catchment_extended_rows/wais_coastline_all_32m_pig_catchment_extended_rows_20151030_tocut_lowcount_name.txt

stereo_dir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment_extended_rows/stereo_ln
stereo_fn=''
prefix_list=$(ls $stereo_dir/*DEM_32m.tif | awk -F'-' '{print $1}' | sort -u)
#prefix_list=$(ls $stereo_dir/*DEM_32m.tif | awk -F'-' '{print $1}' | sort -u | head -1)
for i in $prefix_list 
do
    if cat $stereo_list_tocut $stereo_list_tocut_lowcount | grep -q $(basename $i) ; then
        echo "Skipping $i"
    else
        #if [ -e ${i}-DEM_32m_trans.tif ] ; then
        #if [ -e ${i}-DEM_32m_trans_tilt_removed-adj.tif ] ; then
        #    stereo_fn+=" ${i}-DEM_32m_trans_tilt_removed-adj.tif"
        #if [ -e ${i}-DEM_32m_trans.tif ] ; then
        #    stereo_fn+=" ${i}-DEM_32m_trans.tif"
        #    #stereo_fn+=" ${i}-DEM_32m_trans-adj.tif"
        #else
            stereo_fn+=" ${i}-DEM_32m.tif"
            #stereo_fn+=" ${i}-DEM_32m-adj.tif"
        #fi
    fi
done
echo
echo $(echo $stereo_fn | wc -w) WV stereo
echo $stereo_fn | tr ' ' '\n'
echo
fn+=" $stereo_fn"

echo $(echo $fn | wc -w) Total
#echo $fn
echo

#res=256
res=256
extent=''
#This is padded shelf
#extent='-1692221 -365223 -1551556 -245479'
#This is tworows with long extent up catchment
#extent='-1741216 -380432 -1507783 -38064'

#Extent for tworows, full width, shelf and upstream ~50 km - use for tiltcorr
extent='-1747550 -366842 -1488360 -125048' 
#Apply tiltcorr
#Extent for tworows, clipped to area of interest
#extent='-1705940 -362954 -1515208 -209235'
#Extent for shelf only, clipped

#This is the main shelf extent (used for detailed flux gate analysis)
#extent='-1627608 -326122 -1569418 -244119'
#This is shelf_clip
#extent='-1694558.0 -366968.0 -1552734.0 -245624.0'

#Extent for GL paper
#extent='-1615158 -300720 -1567987 -229925'
#Ian's GL paper extent
#extent='-1625000 -305000 -1575000 -235000'

#GPS time series extent
#extent='-1611996 -310692 -1603369 -297392'

#outdir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment_extended_rows/pig_stack_20151201
#outdir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment_extended_rows/pig_stack_20151201_origSPIRIT
#outdir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment_extended_rows/pig_stack_20151201_origSPIRIT_20090103corr
#outdir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment_extended_rows/pig_stack_20151201_shelfonly_ptdb64m
#outdir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment_extended_rows/pig_stack_20151201_tworows
#outdir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment_extended_rows/pig_stack_20151201_tworows_oldtiltcorr_15iter
#outdir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment_extended_rows/pig_stack_20151201_shelf_oldtiltcorr_15iter
#outdir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment_extended_rows/pig_stack_20151201_tworows_highcount
outdir=/nobackupp8/deshean/wais_coastline/dem/pig_catchment_extended_rows/pig_stack_20151201_tworows_highcount_20160307/noICP

if [ ! -d $outdir ] ; then
    mkdir $outdir
fi

make_stack.py --no-med -tr $res -te "$extent" -outdir $outdir $fn

exit

cd $outdir
for i in $fn
do
    ln -v -s ../$i .
done

#fn_rel=$(echo $fn | tr ' ' '\n' | awk -F'/' '{print $NF}')
#stack_movie.sh $fn_rel

