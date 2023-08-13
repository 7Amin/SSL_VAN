#!/bin/bash
#SBATCH --job-name=MSD_PRE
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
task=$2
freeze=$3


if [ $job -eq 4127 ] && [ $freeze -eq 1 ]
then
  PYTHONPATH=. python3 MSD/main.py --use_ssl_pretrained --workers 8  --task $task  --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json  --logdir ./runs/MSD_new/test_log80 --save_checkpoint --max_epochs 50 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --freeze yes  \
  --valid_loader valid_loader --model_v VANV412 --upsample vae --lrschedule none --optim_lr 0.0002 --checkpoint \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV412_80_500_128_True_5_pre_version2__final.pt

elif [ $job -eq 41227 ] && [ $freeze -eq 1 ]
then
  PYTHONPATH=. python3 MSD/main.py --use_ssl_pretrained --workers 8  --task $task  --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json  --logdir ./runs/MSD_new/test_log80 --save_checkpoint --max_epochs 50 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --freeze yes  \
  --valid_loader valid_loader --model_v VANV4122GL --upsample vae --lrschedule none --optim_lr 0.0002 --checkpoint \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV4121GL_2_80_500_128_True_5_pre_version2__final.pt

elif [ $job -eq 4127 ] && [ $freeze -eq 0 ]
then
  PYTHONPATH=. python3 MSD/main.py --use_ssl_pretrained --workers 8  --task $task  --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json  --logdir ./runs/MSD_new/test_log80 --save_checkpoint --max_epochs 1000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --freeze no  \
  --valid_loader valid_loader --model_v VANV412 --upsample vae --lrschedule none --optim_lr 0.0001 --checkpoint \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV412_80_500_128_True_5_pre_version2__final.pt

elif [ $job -eq 41227 ] && [ $freeze -eq 0 ]
then
  PYTHONPATH=. python3 MSD/main.py --use_ssl_pretrained --workers 8  --task $task  --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json  --logdir ./runs/MSD_new/test_log80 --save_checkpoint --max_epochs 1000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --freeze no  \
  --valid_loader valid_loader --model_v VANV4122GL --upsample vae --lrschedule none --optim_lr 0.0001 --checkpoint \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV4121GL_2_80_500_128_True_5_pre_version2__final.pt
fi