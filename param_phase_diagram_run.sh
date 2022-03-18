#!/bin/bash
#SBATCH --job-name=param_phase_diagram   # Job name	
#SBATCH --partition cpunodes
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --mem=1gb                   # Job Memory
#SBATCH --time=00:05:00             # Time limit hrs:min:sec
#SBATCH --output=param_phase_diagram_log/%a.log
#SBATCH --array=1-5                # Array range

source activate jax
python param_phase_diagram.py $SLURM_ARRAY_TASK_ID