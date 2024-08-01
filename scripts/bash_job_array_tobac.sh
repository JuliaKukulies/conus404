#!/bin/bash -l
#PBS -N convection_tracking_conus
#PBS -A P54048000
### Each array subjob will be assigned a single CPU with x GB of memory
#PBS -l select=1:ncpus=1:mem=50GB
#PBS -l walltime=12:00:00
#PBS -q casper
### Request 10 subjobs with array indices spanning 2002-2003 (input year)
#PBS -J 2018-2020
#PBS -j oe
#PBS -r y
#PBS -o log_files/tobac_tracking.log

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

### Run program
/glade/work/kukulies/conda-envs/tobac/bin/python convection_tracking_tobac.py $PBS_ARRAY_INDEX
