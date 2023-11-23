#!/bin/bash
#SBATCH -A geomofdata
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -t 35:00:00
#SBATCH --mem=64000

REPO_ROOT=/scratch/ejg8qa/MLIA-2022-final

module load singularity pytorch/1.12.0
#export PYTHONPATH=/scratch/ejg8qa/MONAI:/scratch/ejg8qa/MLIA-2022-final/src

singularity run $CONTAINERDIR/pytorch-1.12.0.sif -m pip install $REPO_ROOT
singularity run --nv $CONTAINERDIR/pytorch-1.12.0.sif \
$REPO_ROOT/src/swin/MLIA-main.py --train \
--net-cfg $REPO_ROOT/configs/basic.json \
--input /scratch/ejg8qa/CardiacImage_data/Training \
--output /scratch/ejg8qa/9_layer2
