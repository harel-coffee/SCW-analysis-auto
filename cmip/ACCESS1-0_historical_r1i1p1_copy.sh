#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=06:00:00,mem=190GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/wrf_python_ACCESS1-0_historical_reduced.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/wrf_python_ACCESS1-0_historical_reduced.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38
 
#Set up conda/shell environments 
source activate wrfpython3.6 

#d=1960-01-01
#while [ "$d" != 2010-01-01 ]; do

#	start_time=$(date -d "$d" +%Y)"010100"
#	end_time=$(date -d "$d + 4 year"  +%Y)"123118"
#
#	python /home/548/ab4502/working/ExtremeWind/wrf_non_parallel.py -m ACCESS1-0 -r aus -t1 $start_time -t2 $end_time --issave True --outname ACCESS1-0_historical_r1i1p1 -e historical --ens r1i1p1

#	d=$(date -I -d "$d + 5 year")

#done

python /home/548/ab4502/working/ExtremeWind/wrf_non_parallel_reduced.py -m ACCESS1-0 -r global -t1 1960010100 -t2 1960123118 --issave True --outname ACCESS1-0_historical_r1i1p1_reduced -e historical --ens r1i1p1