#!/bin/bash
#PBS -N C404_PGW 
#PBS -A P66770001
### Each array subjob will be assigned a single CPU with x GB of memory
#PBS -l select=1:ncpus=1:mem=200GB
#PBS -l walltime=24:00:00
#PBS -q casper
### Request 10 subjobs with array indices spanning, e.g. 2000-2022 (input year)
#PBS -J 2000-2022
#PBS -j oe
#PBS -r y
#PBS -o log_files/conus_pgw_processing.log
#PBS -m n

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR
echo $PBS_ARRAY_INDEX

### Run program
/glade/work/kukulies/conda-envs/iwp_preprocessing/bin/python postprocessing_variables_conus404_pgw.py $PBS_ARRAY_INDEX 12
