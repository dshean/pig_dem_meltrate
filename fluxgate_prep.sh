#! /bin/bash

#Extract flux gates
mb_dir=/scr/pig_stack_20160307_tworows_highcount/flux_gates/test
mkdir $mb_dir
cd $mb_dir

#Need to do surface, bed, vx, vy
main_shp_fn=/scr/pig_stack_20160307_tworows_highcount/flux_gates/pig_flux_gates_20151204.shp
ns_shp_fn=/scr/pig_stack_20160307_tworows_highcount/flux_gates/pig_flux_gates_NSshelf_highcount_20151207.shp

for fn in $main_shp_fn $ns_shp_fn
do
    cp -v ${fn%.*}.* $mb_dir
done

#dem_stack_fn=/scr/pig_stack_20160307_tworows_highcount/full_extent/new_workflow/more_testing/20021128_2050_atm_256m-DEM_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans_stack_730_tide_ib_mdt_geoid_removed_nocorr_offset_-3.10m_filt_DG+LVIS+ATM_2009-2016_lsq_tiltcorr_filt_nocorrinv_GLAS+SPIRIT_merge_extract/mos_year/20080101_20070701-20080630_mos-tile-0_20150101_20140701-20150630_mos-tile-0_stack_8_LinearNDint_237_253_365day.npz
dem_stack_fn=/scr/pig_stack_20160307_tworows_highcount/full_extent/new_workflow/more_testing/20021128_2050_atm_256m-DEM_20150406_1519_103001003F89B700_1030010040D68600-DEM_32m_trans_stack_730_tide_ib_mdt_geoid_removed_nocorr_offset_-3.10m_filt_DG+LVIS+ATM_2009-2016_lsq_tiltcorr_filt_nocorrinv_GLAS+SPIRIT_merge_extract/mos_year/20080101_20070701-20080630_mos-tile-0_20150101_20140701-20150630_mos-tile-0_stack_8_LinearNDint_473_506_365day.npz
vx_stack_fn=/scr/pig_stack_20151201_tworows_highcount/vel_20160225_512m/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vx_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vx_stack_22_clip_LinearNDint_237_277_121day.npz
vy_stack_fn=/scr/pig_stack_20151201_tworows_highcount/vel_20160225_512m/20060921_1746_189days_20060521_0950-20061126_0822_alos_mos_Track-Pig06_vy_20151109_1349_90days_20151014_0000-20160112_0000_ls8_mos_All_S1516_vy_stack_22_clip_LinearNDint_237_277_121day.npz
#bed_stack_fn=/Volumes/insar3/dshean/ben_pig_bed_2016/pig_bed_2016_bsmith_aniso-adj_pig_bed_2016_bsmith_aniso-adj_stack_1.npz
#bed_stack_fn=/Volumes/insar3/dshean/ben_pig_bed_2016/pig_bed_2016_bsmith_aniso-adj_embed_bedmap2_pig_bed_2016_bsmith_aniso-adj_embed_bedmap2_stack_1.npz
bed_stack_fn=/Volumes/insar3/dshean/ben_pig_bed_2016/pig_bed_2016_bsmith_aniso_v2-adj_pig_bed_2016_bsmith_aniso_v2-adj_stack_1.npz

for fn in $dem_stack_fn $vx_stack_fn $vy_stack_fn $bed_stack_fn
do
    cp -v $fn $mb_dir
done

echo

main_shp_fn=$(basename $main_shp_fn)
ns_shp_fn=$(basename $ns_shp_fn)
dem_stack_fn=$(basename $dem_stack_fn)
vx_stack_fn=$(basename $vx_stack_fn)
vy_stack_fn=$(basename $vy_stack_fn)
bed_stack_fn=$(basename $bed_stack_fn)

ln -sf $dem_stack_fn dem.npz
ln -sf $vx_stack_fn vx.npz
ln -sf $vy_stack_fn vy.npz
ln -sf $bed_stack_fn bed.npz

for shp in $main_shp_fn $ns_shp_fn
do
    #for stack in $dem_stack_fn $vx_stack_fn $vy_stack_fn $bed_stack_fn
    for stack in dem.npz vx.npz vy.npz bed.npz
    do
        extract_profile.py $shp $stack 
    done
done

#Run discharge

#Run massbudget
