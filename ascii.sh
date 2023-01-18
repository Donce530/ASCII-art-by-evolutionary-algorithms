#!/bin/bash
#SBATCH -J hello_world
#SBATCH --partition=main
#SBATCH -t 20:00:00
#SBATCH --cpus-per-task=100
#SBATCH --mem=50000

# your code goes below
module load any/python/3.8.3-conda
conda activate ascii

python ascii_evolution.py