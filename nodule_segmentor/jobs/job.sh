#!/bin/bash
#SBATCH --job-name=segmentation
#SBATCH --time=01:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH -p devel
#SBATCH --gpus-per-node=1
#SBATCH --mem=40gb

cd SSL_VAN
module spider cuda
conda activate ssl_van_seg
#pip3 install -r ./requirements.txt

PYTHONPATH=. python3 nodule_segmentor/main.py --num_workers 8 \
 --batch_size 8 --base_data ../images --luna_data /luna16 --base_dir_code ./ --patch_size 96

