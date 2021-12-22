#!/bin/bash
# Submission script for tesla
#SBATCH --partition=tesla
#SBATCH --gres=gpu:1
#SBATCH --job-name=training 
#SBATCH --time=10-01:00:00 # days-hh:mm:ss
##SBATCH --nodelist=alan-compute-13
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/home/mvasist/scripts_new/output/16_params/training_out_5M_-%A-%a.txt

conda activate petitRT
cd /home/mvasist/scripts_new/parallel/
python training_ViT.py 