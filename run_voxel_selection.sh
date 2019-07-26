#!/usr/bin/env bash
# Input python command to be submitted as a job
#
# Specify params
#SBATCH --output=searchlight-%j.out
#SBATCH --job-name af_sl_srun
#SBATCH -p short
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

SUBJECT=$1
LAYER=$2
CONTEXT=$3
FOLDID=$4

# Run the python script
srun --mpi=pmi2  python voxel_selection.py $SUBJECT $LAYER $CONTEXT $FOLDID
