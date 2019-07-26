#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=svm
#SBATCH --output=svm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=6000 
#SBATCH --time=5:00:00

n=$1
layer=$2
sub=$3

module load Apps/R
Rscript svm_classification.R $n $layer $sub
