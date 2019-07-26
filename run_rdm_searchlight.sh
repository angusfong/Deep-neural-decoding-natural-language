#!/usr/bin/env bash
# Input python command to be submitted as a job
#
# Specify params
#SBATCH --output=searchlight-%j.out
#SBATCH --job-name rdm-searchlight
#SBATCH -p short
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

SUBJECT=$1
CONTEXT=$2

# Run the python script
srun --mpi=pmi2  python rdm_searchlight.py $SUBJECT $CONTEXT
