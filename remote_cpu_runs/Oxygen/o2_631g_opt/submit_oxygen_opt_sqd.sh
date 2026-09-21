#!/bin/bash
#SBATCH --job-name=oxygen_opt_sqd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=tcp-normal
#SBATCH --exclude=node5
#SBATCH --time=04:00:00
#SBATCH --output=oxygen_opt_sqd_%j.out
#SBATCH --error=oxygen_opt_sqd_%j.err

date
ulimit -s unlimited
ulimit -l unlimited

# Module load is required INSIDE the job too — it does not carry over from the login shell.
module load anaconda3/anaconda3-python3.9.13
source ~/sqd-env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python3 oxygen_opt_sqd_cluster.py

date
