#!/bin/bash

for i in $(seq 1980 1 2018); do
 
 let j=i+1

 cp /home/548/ab4502/working/ExtremeWind/jobs/merra_wrfpython/wrfpython_nonparallel_merra_generic.sh /home/548/ab4502/working/ExtremeWind/jobs/merra_wrfpython/wrfpython_nonparallel_merra_$i.sh

 sed -i "s/YEAR/$i/g" /home/548/ab4502/working/ExtremeWind/jobs/merra_wrfpython/wrfpython_nonparallel_merra_$i.sh
 sed -i "s/YearPlusOne/$j/g" /home/548/ab4502/working/ExtremeWind/jobs/merra_wrfpython/wrfpython_nonparallel_merra_$i.sh

 qsub /home/548/ab4502/working/ExtremeWind/jobs/merra_wrfpython/wrfpython_nonparallel_merra_$i.sh

 done


