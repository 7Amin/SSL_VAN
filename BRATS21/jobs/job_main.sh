#!/bin/bash
#SBATCH --job-name=segmentation
#SBATCH --time=23:59:59
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH -p a100
#SBATCH --gpus-per-node=2
#SBATCH --mem=80gb

cd SSL_VAN
module spider cuda
conda activate ssl_van_seg
#pip3 install -r ./requirements.txt

PYTHONPATH=. python3 BRATS21/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
--json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 10000 \
--distributed --use_normal_dataset --batch_size 8 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
--mlp_ratios 8 8 4 4 --checkpoint --val_every 5
