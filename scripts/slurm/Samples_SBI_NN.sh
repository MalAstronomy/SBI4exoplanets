#!/bin/bash
# Submission script for any
#SBATCH --job-name=sampltesla
#SBATCH --partition=tesla
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00 
#SBATCH --mem-per-cpu=18G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/mvasist/scripts_new/output/16_params/samples_tesla_mod-%A.txt

cd /home/mvasist/scripts_new/parallel/
python Samples_SBI_NN.py
echo 'done!'

