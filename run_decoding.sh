#!/usr/bin/env bash
# Input python command to be submitted as a job
#
# Specify params
#SBATCH --output=decoding-%j.out
#SBATCH --job-name af_sl_srun
#SBATCH -p short
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

SUBJECT=$1
LAYER=$2

# Run the python script
srun --mpi=pmi2  python decoding.py $SUBJECT $LAYER
