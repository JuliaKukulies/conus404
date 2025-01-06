#!/bin/bash
#PBS -N moaap_ar
#PBS -A P54048000
### Each array subjob will be assigned a single CPU with x GB of memory
#PBS -l select=1:ncpus=1:mem=40GB
#PBS -l walltime=24:00:00
#PBS -q casper
### Request 10 subjobs with array indices spanning, e.g. 2000-2022 (input year)
#PBS -J 2000-2022
#PBS -j oe
#PBS -r y
#PBS -o log_files/moaap_features_2000_2022_AR.log
#PBS -m n 
export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR
echo $PBS_ARRAY_INDEX

### Run program
/glade/work/kukulies/conda-envs/tobac/bin/python get_statistics_moaap_features.py AR $PBS_ARRAY_INDEX
