#!/bin/bash
#PBS -N conus404 
#PBS -A P54048000
### Each array subjob will be assigned a single CPU with x GB of memory
#PBS -l select=1:ncpus=1:mem=200GB
#PBS -l walltime=24:00:00
#PBS -q casper
### Request 10 subjobs with array indices spanning, e.g. 2000-2022 (input year)
#PBS -J 2003-2005
#PBS -j oe
#PBS -r y
#PBS -o log_files/conus_processing.log

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR
echo $PBS_ARRAY_INDEX

### Run program
/glade/work/kukulies/conda-envs/iwp_preprocessing/bin/python postprocessing_variables_conus404_correction.py $PBS_ARRAY_INDEX
