#!/bin/bash
#SBATCH --job-name=PreT1
#SBATCH --time=05:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -p a100
#SBATCH --gpus-per-node=2
#SBATCH -G 2
#SBATCH --mem=80gb

module spider cuda
conda activate ssl_van_seg
#pip3 install -r ./requirements.txt


job=$1

if [ $job -eq 47 ]
then
  PYTHONPATH=. python3 pretrain/main.py --mode server  --workers 8 --logdir ./runs/pre_train_1/test_log \
  --save_checkpoint --max_epochs 100 --use_normal_dataset --batch_size 1 --num_stages 4 \
  --embed_dims 64 128 256 512 --depths 3 4 6 3 --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 \
  --upsample vae  --checkpoint --model_inferer inferer --valid_loader valid_loader --model_v PREVANV4

elif [ $job -eq 427 ]
then
PYTHONPATH=. python3 pretrain/main.py --mode server  --workers 8 --logdir ./runs/pre_train_1/test_log \
  --save_checkpoint --max_epochs 100 --use_normal_dataset --batch_size 1 --num_stages 4 \
  --embed_dims 64 128 256 512 --depths 3 4 6 3 --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 \
  --upsample vae  --checkpoint --model_inferer inferer --valid_loader valid_loader --model_v PREVANV4GL

elif [ $job -eq 4127 ]
then
PYTHONPATH=. python3 pretrain/main.py --mode server  --workers 8 --logdir ./runs/pre_train_1/test_log \
  --save_checkpoint --max_epochs 100 --use_normal_dataset --batch_size 1 --num_stages 4 \
  --embed_dims 64 128 256 512 --depths 3 4 6 3 --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 \
  --upsample vae  --checkpoint --model_inferer inferer --valid_loader valid_loader --model_v PREVANV412

elif [ $job -eq 41217 ]
then
PYTHONPATH=. python3 pretrain/main.py --mode server  --workers 8 --logdir ./runs/pre_train_1/test_log \
  --save_checkpoint --max_epochs 100 --use_normal_dataset --batch_size 1 --num_stages 4 \
  --embed_dims 64 128 256 512 --depths 3 4 6 3 --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 \
  --upsample vae  --checkpoint --model_inferer inferer --valid_loader valid_loader --model_v PREVANV4121GL

elif [ $job -eq 527 ]
then
PYTHONPATH=. python3 pretrain/main.py --mode server  --workers 8 --logdir ./runs/pre_train_1/test_log \
  --save_checkpoint --max_epochs 100 --use_normal_dataset --batch_size 1 --num_stages 4 \
  --embed_dims 64 128 256 512 --depths 3 4 6 3 --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 \
  --upsample vae  --checkpoint --model_inferer inferer --valid_loader valid_loader --model_v PREVANV5GL

elif [ $job -eq 627 ]
then
PYTHONPATH=. python3 pretrain/main.py --mode server  --workers 8 --logdir ./runs/pre_train_1/test_log \
  --save_checkpoint --max_epochs 100 --use_normal_dataset --batch_size 1 --num_stages 4 \
  --embed_dims 64 128 256 512 --depths 3 4 6 3 --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 \
  --upsample vae  --checkpoint --model_inferer inferer --valid_loader valid_loader --model_v PREVANV6GL

elif [ $job -eq 49 ]
then
  PYTHONPATH=. python3 pretrain/main.py --mode server  --workers 8 --logdir ./runs/pre_train_1/test_log \
  --save_checkpoint --max_epochs 100 --use_normal_dataset --batch_size 1 --num_stages 4 \
  --embed_dims 96 192 384 768 --depths 3 3 24 3 --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 \
  --upsample vae  --checkpoint --model_inferer inferer --valid_loader valid_loader --model_v PREVANV4

elif [ $job -eq 429 ]
then
PYTHONPATH=. python3 pretrain/main.py --mode server  --workers 8 --logdir ./runs/pre_train_1/test_log \
  --save_checkpoint --max_epochs 100 --use_normal_dataset --batch_size 1 --num_stages 4 \
  --embed_dims 96 192 384 768 --depths 3 3 24 3 --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 \
  --upsample vae  --checkpoint --model_inferer inferer --valid_loader valid_loader --model_v PREVANV4GL

elif [ $job -eq 4129 ]
then
PYTHONPATH=. python3 pretrain/main.py --mode server  --workers 8 --logdir ./runs/pre_train_1/test_log \
  --save_checkpoint --max_epochs 100 --use_normal_dataset --batch_size 1 --num_stages 4 \
  --embed_dims 96 192 384 768 --depths 3 3 24 3 --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 \
  --upsample vae  --checkpoint --model_inferer inferer --valid_loader valid_loader --model_v PREVANV412

elif [ $job -eq 41219 ]
then
PYTHONPATH=. python3 pretrain/main.py --mode server  --workers 8 --logdir ./runs/pre_train_1/test_log \
  --save_checkpoint --max_epochs 100 --use_normal_dataset --batch_size 1 --num_stages 4 \
  --embed_dims 96 192 384 768 --depths 3 3 24 3 --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 \
  --upsample vae  --checkpoint --model_inferer inferer --valid_loader valid_loader --model_v PREVANV4121GL

elif [ $job -eq 529 ]
then
PYTHONPATH=. python3 pretrain/main.py --mode server  --workers 8 --logdir ./runs/pre_train_1/test_log \
  --save_checkpoint --max_epochs 100 --use_normal_dataset --batch_size 1 --num_stages 4 \
  --embed_dims 96 192 384 768 --depths 3 3 24 3 --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 \
  --upsample vae  --checkpoint --model_inferer inferer --valid_loader valid_loader --model_v PREVANV5GL

elif [ $job -eq 629 ]
then
PYTHONPATH=. python3 pretrain/main.py --mode server  --workers 8 --logdir ./runs/pre_train_1/test_log \
  --save_checkpoint --max_epochs 100 --use_normal_dataset --batch_size 1 --num_stages 4 \
  --embed_dims 96 192 384 768 --depths 3 3 24 3 --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 \
  --upsample vae  --checkpoint --model_inferer inferer --valid_loader valid_loader --model_v PREVANV6GL
fi