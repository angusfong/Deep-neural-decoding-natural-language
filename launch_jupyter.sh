#!/bin/bash
# Launch a jupyter notebook on a node and output the necessary details to put in to your browser

# Take in a partition name if provided
if [ $# -gt 0 ]
then
partition=$1
else
partition=short
fi

# Check that you don't already have a tunnel set up
job_number=`squeue -u $USER | grep tunnel | wc -l`

if [ $job_number -gt 4 ]
then
	echo "You have $job_number tunnel jobs running. This is too many, you cannot make a new tunnel."
	exit

elif [ $job_number -gt 0 ]
then
	job_report=`squeue -u $USER | grep tunnel`
	echo "You have $job_number tunnel jobs running. You could use an alternative job id:"
	echo "$job_report"
	
	# What for a key press to continue
	sleep 1
	read -p "Press any key to continue, press ctrl + c to quit " -n1 -s
	sleep 1
fi

# Submit the job
jobid=`sbatch -p $partition ./run_jupyter.sh`

# Pull out just the id
jobid=${jobid/Submitted batch job }

# Wait for the job file to be created and when it is print
while [ ! -e ./jupyter-log-${jobid}.txt ]
do
  sleep 1s
done

# Once the file is created, wait for it to have printed at least 10 lines
line_number=`cat ./jupyter-log-${jobid}.txt | wc -l`
while [ $line_number -lt 15 ]
do
  sleep 1s;
  line_number=`cat ./jupyter-log-${jobid}.txt | wc -l`
done

# Print the script once it exists
head -20 ./jupyter-log-${jobid}.txt

# Warn user
echo "##### Be aware, job should be terminated with scancel $jobid when you have finished with the notebook ######"
