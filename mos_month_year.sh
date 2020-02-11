#! /bin/bash

#David Shean
#dshean@gmail.com

#Utility to mosaic input DEMs by year and month

ulimit -n 9999

#ext='DEM_tr16x.tif'
#mkdir mos; cd mos
#for i in ../WV*/dem*/*DEM_tr16x.tif; do ln -s $i .; done
#parallel -j 32 'gdalwarp -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER -tr 32 32 -r cubic -dstnodata 0 -overwrite {} {.}_32m.tif' ::: *${ext}

#Note: all of this should be parallelized
#res=32
#ext='DEM_average_16x.tif'
#ext='DEM_tr16x_32m.tif'
#ext='DEM_32m.tif'
#ext='32m.tif'
#ext='32m_trans.tif'
#ext='meltrate_init.tif'
#ext='meltrate.tif'
#ext='stack_med.tif'
ext='.tif'
#ext="{_3[56789]*day_*rate.tif,_4*day*rate.tif,_5*day*rate.tif}"
#ext='meltrate_mid.tif'
njobs=4
year=true
season=false
month=false

gdal_opt='-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER'

date
echo

#list4x=($(ls WV*/dem*/*${ext}))
#years=$(ls WV*/dem*/*${ext} | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}' | cut -c 1-4 | sort -n -u)
#months=$(ls WV*/dem*/*${ext} | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}' | cut -c 1-6 | sort -n -u)

list4x=($(eval ls *${ext}))
years=$(eval ls *${ext} | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}' | cut -c 1-4 | sort -n -u)
#Want to subtract one from first entry 
years='2007 2008 2009 2010 2011 2012 2013 2014 2015'
#years="2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016"
months=$(eval ls *${ext} | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}' | cut -c 1-6 | sort -n -u)

#list4x=($(ls *align/*${ext}))
#years=$(ls *align/*${ext} | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}' | cut -c 1-4 | sort -n -u)
#months=$(ls *align/*${ext} | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}' | cut -c 1-6 | sort -n -u)

proj="$(proj_select.py ${list4x[0]})"

echo $proj
echo

function makemos () {
    t1_t2=$1
    #This is an ugly, ugly hack
    t1=$(echo $t1_t2 | sed 's/\\//' | awk '{print $1}')
    t2=$(echo $t1_t2 | sed 's/\\//' | awk '{print $2}')
    ct=$(echo $t1_t2 | sed 's/\\//' | awk '{print $3}')
    echo $t1
    echo $t2
    echo $ct
    #t1=$1
    #t2=$2
    outdir=$2
    inlist=$(echo $3 | sed "s/'//g")
    #if (( $t1 >> $t2 )) ; then
    #    echo "$t1 is greater than $t2"
    #else
        mylist=()
        for i in $inlist
        do
            t=$(echo $i | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}')
            if (( $t >= $t1 )) && (( $t <= $t2 )) ; then
                mylist+=($i)
            fi
        done
        mylist=${mylist[@]}
        n=$(echo $mylist | wc -w)
        if (( $n > 0 )) ; then 
            #outmos="$outdir/${t1}-${t2}_mos_order.tif"
            outmos="$outdir/${ct}_${t1}-${t2}_mos"
            echo 
            echo "$n $outmos"
            echo
            echo $mylist | tr ' ' '\n'
            echo
            if [ ! -d $outdir ] ; then
                mkdir $outdir
            fi
            #if [ ! -e $outmos ] ; then
                #gdalwarp -overwrite -tr $res $res $gdal_opt -r cubic -dstnodata 0 $mylist $outmos 2>&1 | tee ${outmos%%.*}.log 
                #mos_cluster.py -o $outmos $mylist &> ${outmos%%.*}.log
                #mos_cluster.py -o $outmos $mylist | tee ${outmos%%.*}.log
                n_threads=4
                dem_mosaic --threads $n_threads -o $outmos $mylist | tee ${outmos%%.*}.log
                dem_mosaic --threads $n_threads --count -o $outmos $mylist | tee ${outmos%%.*}.log
                dem_mosaic --threads $n_threads --stddev -o $outmos $mylist | tee ${outmos%%.*}.log
                dem_mosaic --threads $n_threads --median -o $outmos $mylist | tee ${outmos%%.*}.log
                #dem_mosaic --threads $n_threads --mean -o $outmos $mylist | tee ${outmos%%.*}.log
                #dem_mosaic --threads $n_threads --min -o $outmos $mylist | tee ${outmos%%.*}.log
                #dem_mosaic --threads $n_threads --max -o $outmos $mylist | tee ${outmos%%.*}.log
                dem_mosaic --threads $n_threads --first -o $outmos $mylist | tee ${outmos%%.*}.log
                dem_mosaic --threads $n_threads --last -o $outmos $mylist | tee ${outmos%%.*}.log
                make_stack.py -stack_fn ${outmos}_stack --no-trend $mylist | tee ${outmos%%.*}.log
                #gdaldem hillshade $outmos ${outmos%%.*}_hs.tif 
                #hs.sh ${outmos}-tile-0.tif
                #imview.py -of png -label 'Elevation (m WGS84)' -overlay ${outmos%%.*}_hs.tif $outmos
            #fi
        fi
    #fi
    echo
}
export -f makemos

function compute_dh () {
    outdir=$1
    pushd $outdir
    #list=($(ls *mos_${res}m.tif))
    list=($(ls *mos_order.tif))
    if (( ${#list[@]} > 1 )) ; then
        for i in $(seq 0 $((${#list[@]}-2)))
        do 
            echo compute_dh.py ${list[i]} ${list[((i+1))]}
            compute_dh.py ${list[i]} ${list[((i+1))]} &
            echo
        done
        compute_dh.py ${list[0]} ${list[((i+1))]}
    fi
    pushd
}
export -f compute_dh

if $month ; then
    outdir=mos_month
    for i in $months
    do
        t1=${i}01
        t2=${i}31
        makemos $t1 $t2 $outdir
    done
    wait; date
    compute_dh $outdir
    wait; date 
fi

#Probably want to add support for winter/summer or seasonal mosaics here
#UTM Zone 10N, PNW
#if $(echo $proj | grep -q '+proj=utm +zone=10') ; then
#    t1=${i}1001
#    t2=$((i+1))0531
#t1=${i}1001
#t2=$((i+1))0531
#t1=${i}0601
#t2=$((i+1))0930

#Create yearly from monthly
#More efficient?

if $season ; then
    outdir=mos_seasonal
    arglist=''
    for i in $years
    do
        t1=${i}0101
        t2=${i}0531
        ct=${i}0315
        #t1=${i}1222
        #t2=$((i+1))0621
        #arglist+=\ \""$t1 $t2"\" 
        arglist+=\ \""$t1 $t2 $ct"\" 
        t1=${i}0601
        t2=${i}1031
        ct=${i}0815
        #t1=${i}0622
        #t2=${i}1222
        #t1=${i}0601
        #t2=${i}1231
        arglist+=\ \""$t1 $t2 $ct"\" 
    done
    echo $arglist
    eval parallel -v -j $njobs "makemos {} $outdir \'${list4x[@]}\'" ::: ${arglist[@]}
    #makemos "20120601 20130531" $outdir \'${list4x[@]}\'
    wait; date
    #compute_dh $outdir
    #wait; date
fi

if $year ; then
    outdir=mos_year
    arglist=''
    for i in $years
    do
        #Polar stereographic north
        #elif $(echo $proj | grep -q '+proj=stere +lat_0=90') ; then
        #Polar stereographic south
        #if $(echo $proj | grep -q '+proj=stere +lat_0=-90') ; then
            t1=${i}0701
            t2=$((i+1))0630
            ct=$((i+1))0101
        #else
            #t1=${i}0101
            #t2=${i}1231
            #ct=${i}0701
        #fi
        #makemos $t1 $t2 $outdir
        arglist+=\ \""$t1 $t2 $ct"\" 
    done
    echo $arglist
    eval parallel -v -j $njobs "makemos {} $outdir \'${list4x[@]}\'" ::: ${arglist[@]}
    #makemos "20120601 20130531" $outdir \'${list4x[@]}\'
    wait; date
    #compute_dh $outdir
    #wait; date
fi

exit

#Combine yearly for all-inclusive mosaic
cd mos_year
list=($(ls *mos_order.tif))
if (( ${#list[@]} > 1 )) ; then
    gdalwarp -overwrite -tr $res $res $gdal_opt -r cubic -dstnodata 0 ${list[@]} all_mos_${res}m.tif
    gdaldem hillshade $gdal_opt all_mos_${res}m.tif all_mos_${res}m_hs.tif
fi
list=($(ls *mos_order_ts.tif))
if (( ${#list[@]} > 1 )) ; then
    gdalwarp -overwrite -tr $res $res $gdal_opt -r cubic -dstnodata 0 ${list[@]} all_mos_${res}m_ts.tif
fi

#Make sure using union, not intersection
#./lib/warplib.py $outdir/*.tif
