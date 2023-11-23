#!/bin/bash
#SBATCH -A geomofdata
#SBATCH -c 6
#SBATCH -t 1:00:00
#SBATCH --mem=8000

REPO_ROOT=/scratch/ejg8qa/VAR-next

singularity pull library://icaoberg/psc/ffmpeg
singularity run library://icaoberg/psc/ffmpeg ffmpeg -h

