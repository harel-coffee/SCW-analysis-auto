#!/bin/bash

#PBS -P eg3 
#PBS -q hugemem
#PBS -l walltime=48:00:00,mem=512GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/wrf_python_GFDL-ESM2G_historical.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/wrf_python_GFDL-ESM2G_historical.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38+gdata/al33
 
#Set up conda/shell environments 
source activate wrfpython3.6 

d=1960-01-01
while [ "$d" != 2010-01-01 ]; do

	start_time=$(date -d "$d" +%Y)"010100"
	end_time=$(date -d "$d + 9 year"  +%Y)"123118"

	python /home/548/ab4502/working/ExtremeWind/wrf_non_parallel_reduced.py -m GFDL-ESM2G -r global -t1 $start_time -t2 $end_time --issave True --outname GFDL-ESM2G_historical_r1i1p1 -e historical --ens r1i1p1

	d=$(date -I -d "$d + 10 year")

done
