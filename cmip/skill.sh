#!/bin/bash

#PBS -P eg3
#PBS -q express
#PBS -l walltime=06:00:00,mem=32GB
#PBS -l ncpus=16
#PBS -o /home/548/ab4502/working/ExtremeWind/jobs/messages/skill.o
#PBS -e /home/548/ab4502/working/ExtremeWind/jobs/messages/skill.e
#PBS -l storage=gdata/eg3+gdata/ub4+gdata/ma05+gdata/du7+gdata/rr3+gdata/r87+gdata/fs38

source activate wrfpython3.6

python /home/548/ab4502/working/ExtremeWind/cmip/skill.py
