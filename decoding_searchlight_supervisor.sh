#!/usr/bin/env bash

# What method are you using to compare the resampled SFNR to?
subjects="1 2 3 4 5 6 7 8"
layers="layer1 glove"

for subject in $subjects
do

for layer in $layers
do

sbatch run_decoding_searchlight.sh $subject $layer

done
done
