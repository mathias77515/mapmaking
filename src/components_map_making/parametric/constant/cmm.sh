#!/bin/bash

#SBATCH --job-name=both_101

# we ask for n MPI tasks with N cores each on c nodes

#SBATCH --partition=hpc
#SBATCH --nodes=1                # c
#SBATCH --ntasks-per-node=3      # n
#SBATCH --cpus-per-task=3        # N
#SBATCH --mem=20G
#SBATCH --time=0-00:40:00
#SBATCH --output=mulitple_jobs_%j.log

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load mpich

mpirun -np $SLURM_NTASKS python cmm_mpi.py $1 $2
