#!/bin/bash -l
#PBS -N convection_tracking_conus
#PBS -A P54058000
### Each array subjob will be assigned a single CPU with x GB of memory
#PBS -l select=1:ncpus=1:mem=50GB
#PBS -l walltime=12:00:00
#PBS -q casper
### Request 10 subjobs with array indices spanning 2002-2003 (input year)
#PBS -J 2002-2003
#PBS -j oe

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

### Run program
/glade/work/kukulies/conda-envs/tobac/bin/python convection_tracking_tobac.py $PBS_ARRAY_INDEX