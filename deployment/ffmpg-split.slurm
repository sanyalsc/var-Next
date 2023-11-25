#!/bin/bash
#SBATCH -A geomofdata
#SBATCH -c 6
#SBATCH -t 1:00:00
#SBATCH --mem=8000
#SBATCH -p standard

DATA_ROOT=/scratch/ejg8qa

singularity pull library://icaoberg/psc/ffmpeg:6.0-r2
singularity exec library://icaoberg/psc/ffmpeg:6.0-r2 ffmpeg -i $REPO_ROOT/log_vids/full_log2_320.mp4 -r 30 -f image2 $REPO_ROOT/log_images_320/img2_%07d.png
#singularity exec library://icaoberg/psc/ffmpeg:6.0-r26 ffmpeg -y -i $REPO_ROOT/log_vids/full_log.mp4 -vf scale=320:-2,setsar=1:1 -c:v libx264 -c:a copy "${REPO_ROOT}/full_log_320.mp4"