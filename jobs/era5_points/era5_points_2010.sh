#!/bin/bash

#PBS -P eg3 
#PBS -q normal
#PBS -l walltime=06:00:00,mem=32GB 
#PBS -l ncpus=1
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_points_2010.o 
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/era5_points_2010.e 
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05
 
#Set up conda/shell environments 
source activate wrfpython3.6 

python /home/548/ab4502/working/ExtremeWind/era5_read.py 2010 2010

