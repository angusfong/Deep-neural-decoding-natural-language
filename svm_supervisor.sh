#!/usr/bin/env bash

layers="glove"
declare -a subs=("1" "2" "3" "4" "5" "6" "7" "8" "AVG")

for n in $(seq 101 195)
do

for layer in $layers
do

for sub in "${subs[@]}"
do

sbatch run_svm_classification.sh $n $layer $sub

done

done

done
