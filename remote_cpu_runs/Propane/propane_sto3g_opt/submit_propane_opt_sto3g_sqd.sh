#!/bin/bash
#SBATCH --job-name=propane_opt_sto3g_sqd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=tcp-normal
#SBATCH --exclude=node5
#SBATCH --time=12:00:00
#SBATCH --output=propane_opt_sto3g_sqd_%j.out
#SBATCH --error=propane_opt_sto3g_sqd_%j.err

# NOTE: --time is 12h (not the 4h used for the diatomics) because propane's
# 20-orbital active space drives the classical diagonalization into millions of
# determinants per iteration. Check `sinfo` for the partition's max time limit
# and raise/lower accordingly. tcp-normal nodes have 8 CPUs each — do not raise
# --cpus-per-task above 8.

date
ulimit -s unlimited
ulimit -l unlimited

# Module load is required INSIDE the job too — it does not carry over from the login shell.
module load anaconda3/anaconda3-python3.9.13
source ~/sqd-env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python3 propane_opt_sto3g_sqd_cluster.py

date
