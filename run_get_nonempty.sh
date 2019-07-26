#!/usr/bin/env bash
# Input python command to be submitted as a job
#
# Specify params
#SBATCH --output=nonempty-%j.out
#SBATCH --job-name nonempty
#SBATCH -p short
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Run the python script
srun  python get_nonempty.py
