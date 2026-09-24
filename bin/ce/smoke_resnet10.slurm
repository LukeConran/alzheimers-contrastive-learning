#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --export=NONE
#SBATCH --job-name=alz_smoke_test_ce
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --gres=gpu:rtx:1
#SBATCH --partition=gpu
#SBATCH --output=./output/smoke_test_ce.%j

module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1 scikit-learn/1.3.1

python train.py \
    --train_json /scratch/group/optmai/chenlang/ADNI/text_data/train/data.json \
    --train_image_dir /scratch/group/optmai/chenlang/ADNI/image_data/train/ \
    --valid_json /scratch/group/optmai/chenlang/ADNI/text_data/validation/data.json \
    --valid_image_dir /scratch/group/optmai/chenlang/ADNI/image_data/validation/ \
    --lr 1e-4 \
    --batch_size 32 \
    --epochs 1 \
    --backbone resnet \
    --model_name resnet10 \
    --checkpoint_pretrain /home/lukeconran/alzheimers/pretrain/resnet_10.pth \
    --label_smoothing 0.1 \
    --output_dir /home/lukeconran/alzheimers/results/resnet10/ce \
    --cache_dir /scratch/user/lukeconran/alz_cache
