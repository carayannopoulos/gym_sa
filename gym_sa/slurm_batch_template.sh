#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --partition={partition}

export PATH="/home/loukas/miniconda3/envs/ML/bin:$PATH"

echo "Starting batch {job_name} at $(date)"
{command}
status=$?
echo "Finished batch {job_name} at $(date) with status $status"
exit $status

