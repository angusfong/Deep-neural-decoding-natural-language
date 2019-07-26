#!/usr/bin/env bash

# What method are you using to compare the resampled SFNR to?
subjects="1 2 3 4 5 6 7 8" 
contexts="0s_ 1s_ 2s_ 4s_ 16s_ 1600s_"

for subject in $subjects
do

for context in $contexts
do

sbatch run_rdm_searchlight.sh $subject $context

done
done
