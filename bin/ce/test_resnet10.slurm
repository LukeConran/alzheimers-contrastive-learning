#!/bin/bash
#SBATCH --export=NONE
#SBATCH --job-name=alz_test_ce
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --gres=gpu:rtx:1
#SBATCH --partition=gpu
#SBATCH --output=./output/test_ce.%j

module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1 scikit-learn/1.3.1

python test.py \
    --test_json /scratch/group/optmai/chenlang/ADNI/text_data/test/data.json \
    --test_image_dir /scratch/group/optmai/chenlang/ADNI/image_data/test/ \
    --checkpoint /home/lukeconran/alzheimers/results/resnet10/ce/best_model_resnet10_bs32.pth \
    --backbone resnet \
    --model_name resnet10 \
    --batch_size 32 \
    --save_preds /home/lukeconran/alzheimers/results/resnet10/ce/preds.npz
