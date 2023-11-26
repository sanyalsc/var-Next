#!/bin/bash
#SBATCH -A geomofdata
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 35:00:00
#SBATCH --mem=64000

REPO_ROOT=/scratch/ejg8qa/var-Next

module load singularity pytorch/2.0.1

singularity run $CONTAINERDIR/pytorch-2.0.1.sif -m pip install $REPO_ROOT
singularity run --nv $CONTAINERDIR/pytorch-2.0.1.sif \
$REPO_ROOT/src/var_next/train_varnext.py \
--net-cfg $REPO_ROOT/configs/2048_latent.json \
--input /scratch/ejg8qa/log_images_360 \
--output /scratch/ejg8qa/vnext_results \
--n-epoch 5
