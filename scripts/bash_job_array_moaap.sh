#!/bin/bash
#PBS -N job_array
#PBS -A P54048000
### Each array subjob will be assigned a single CPU with x GB of memory
#PBS -l select=1:ncpus=1:mem=50GB
#PBS -l walltime=12:00:00
#PBS -q casper
### Request 10 subjobs with array indices spanning 2002-2003 (input year)
#PBS -J 2018-2020
#PBS -j oe
#PBS -r y

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR
echo $PBS_ARRAY_INDEX

### Run program
/glade/work/kukulies/conda-envs/tobac/bin/python get_statistics_moaap_features.py AR $PBS_ARRAY_INDEX
