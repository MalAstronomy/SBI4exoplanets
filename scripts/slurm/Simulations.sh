#!/bin/bash
# Submission script for Hercules2
#SBATCH --job-name=parallel_16params
#SBATCH --array=1-1000 
#SBATCH --time=10-01:00:00 # days-hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2626 # megabytes
#SBATCH --output=/home/mvasist/scripts_new/output/16_params/out_5M_-%A-%a.txt

conda activate petitRT
cd /home/mvasist/scripts_new/parallel/
python Simulations.py $SLURM_ARRAY_TASK_ID