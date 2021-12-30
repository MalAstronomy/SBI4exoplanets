#!/bin/bash
# Submission script for tesla
#SBATCH --partition=tesla
#SBATCH --gres=gpu:1
#SBATCH --job-name=trainnew 
#SBATCH --time=14-00:00:0   # days-hh:mm:ss 10-01:00:00 14-00:00:0
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8GB
#SBATCH --output=/home/mvasist/scripts_new/output/16_params/training_out_5M_new-%A.txt

conda activate petitRT
cd /home/mvasist/scripts_new/parallel/
python training_ViT_new.py 

##SBATCH --nodelist=alan-compute-13
##SBATCH --mem-per-cpu=8G