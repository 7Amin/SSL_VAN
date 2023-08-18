#!/bin/bash
#SBATCH --job-name=BRAPrevae
#SBATCH --time=05:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -p a100
#SBATCH --gpus-per-node=2
#SBATCH -G 2
#SBATCH --mem=80gb

cd SSL_VAN
module spider cuda
conda activate ssl_van_seg
#pip3 install -r ./requirements.txt


job=$1
cluster_size=$2
freeze=$3


if [ $job -eq 4127 ] && [ $cluster_size -eq 80 ] && [ $freeze -eq 1 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice --workers 8 --optim_lr 0.0001 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json  --logdir ./runs/BraTS21_new/test_log --save_checkpoint --max_epochs 1000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer  --lrschedule none --optim_lr 0.0002 \
  --valid_loader valid_loader --model_v VANV412 --upsample vae --checkpoint --use_ssl_pretrained --freeze yes \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV412_80_500_128_True_5_pre_version2__final.pt

elif [ $job -eq 41277 ] && [ $cluster_size -eq 80 ] && [ $freeze -eq 1 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice --workers 8 --optim_lr 0.0001 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json  --logdir ./runs/BraTS21_new/test_log --save_checkpoint --max_epochs 1000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer  --lrschedule none --optim_lr 0.0002 \
  --valid_loader valid_loader --model_v VANV412 --upsample vae --checkpoint --loss DiceCELoss --use_ssl_pretrained --freeze yes \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV412_80_500_128_True_5_pre_version2__final.pt

elif [ $job -eq 41227 ] && [ $cluster_size -eq 80 ] && [ $freeze -eq 1 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice --workers 8 --optim_lr 0.0001 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json  --logdir ./runs/BraTS21_new/test_log --save_checkpoint --max_epochs 1000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer  --lrschedule none --optim_lr 0.0002 \
  --valid_loader valid_loader --model_v VANV4122GL --upsample vae --checkpoint --use_ssl_pretrained --freeze yes \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV4121GL_2_80_500_128_True_5_pre_version2__final.pt


elif [ $job -eq 412277 ] && [ $cluster_size -eq 80 ] && [ $freeze -eq 1 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice --workers 8 --optim_lr 0.0001 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json  --logdir ./runs/BraTS21_new/test_log --save_checkpoint --max_epochs 1000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer  --lrschedule none --optim_lr 0.0002 \
  --valid_loader valid_loader --model_v VANV4122GL --upsample vae --checkpoint --loss DiceCELoss --use_ssl_pretrained --freeze yes \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV4121GL_2_80_500_128_True_5_pre_version2__final.pt


elif [ $job -eq 4127 ] && [ $cluster_size -eq 80 ] && [ $freeze -eq 0 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice --workers 8 --optim_lr 0.0001 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json  --logdir ./runs/BraTS21_new/test_log --save_checkpoint --max_epochs 1000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer  --lrschedule none --optim_lr 0.0001 \
  --valid_loader valid_loader --model_v VANV412 --upsample vae --checkpoint --use_ssl_pretrained --freeze no \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV412_80_500_128_True_5_pre_version2__final.pt

elif [ $job -eq 41277 ] && [ $cluster_size -eq 80 ] && [ $freeze -eq 0 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice --workers 8 --optim_lr 0.0001 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json  --logdir ./runs/BraTS21_new/test_log --save_checkpoint --max_epochs 1000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer  --lrschedule none --optim_lr 0.0001 \
  --valid_loader valid_loader --model_v VANV412 --upsample vae --checkpoint --loss DiceCELoss --use_ssl_pretrained --freeze no \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV412_80_500_128_True_5_pre_version2__final.pt

elif [ $job -eq 41227 ] && [ $cluster_size -eq 80 ] && [ $freeze -eq 0 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice --workers 8 --optim_lr 0.0001 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json  --logdir ./runs/BraTS21_new/test_log --save_checkpoint --max_epochs 1000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer  --lrschedule none --optim_lr 0.0001 \
  --valid_loader valid_loader --model_v VANV4122GL --upsample vae --checkpoint --use_ssl_pretrained --freeze no \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV4121GL_2_80_500_128_True_5_pre_version2__final.pt


elif [ $job -eq 412277 ] && [ $cluster_size -eq 80 ] && [ $freeze -eq 0 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice --workers 8 --optim_lr 0.0001 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json  --logdir ./runs/BraTS21_new/test_log --save_checkpoint --max_epochs 1000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer  --lrschedule none --optim_lr 0.0001 \
  --valid_loader valid_loader --model_v VANV4122GL --upsample vae --checkpoint --loss DiceCELoss --use_ssl_pretrained --freeze no \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV4121GL_2_80_500_128_True_5_pre_version2__final.pt

fi