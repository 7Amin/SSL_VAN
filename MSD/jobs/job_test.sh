#!/bin/bash
#SBATCH --job-name=MSD_TEST
#SBATCH --time=05:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH -p a100
#SBATCH --gpus-per-node=1
#SBATCH -G 1
#SBATCH --mem=160gb

cd SSL_VAN
module spider cuda
conda activate ssl_van_seg
#pip3 install -r ./requirements.txt


job=$1
task=$2


if [ $job -eq 724 ]
then
  PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v SwinUNETR24 --upsample vae --checkpoint --task $task

elif [ $job -eq 736 ]
then
  PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v SwinUNETR36 --upsample vae --checkpoint --task $task

elif [ $job -eq 748 ]
then

  PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v SwinUNETR48 --upsample vae --checkpoint --task $task

elif [ $job -eq 816 ]
then
    PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v UNETR16 --upsample vae --checkpoint --task $task

elif [ $job -eq 832 ]
then
    PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v UNETR32 --upsample vae --checkpoint --task $task

elif [ $job -eq 900 ]
then
  PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v nnUNet --upsample vae --checkpoint --task $task

elif [ $job -eq 1000 ]
then
  PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v SegResNetVAE --upsample vae --checkpoint --task $task

elif [ $job -eq 1100 ]
then
    PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v AttentionUnet --upsample vae --checkpoint --task $task

elif [ $job -eq 1200 ]
then
  PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v Unetpp --upsample vae --checkpoint --task $task

elif [ $job -eq 1300 ]
then
  PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v BasicUNetPlusPlus --upsample vae --checkpoint --task $task

elif [ $job -eq 1410 ]
then
  PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v DiNTS_Search --upsample vae --checkpoint --task $task

elif [ $job -eq 1420 ]
then
  PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v DiNTS_Instance --upsample vae --checkpoint --task $task

elif [ $job -eq 4127 ]
then
  PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV412 --upsample vae --checkpoint --task $task 
  
elif [ $job -eq 41227 ]
then
  PYTHONPATH=. python3 MSD/main.py --test_mode --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 10 \
  --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4122GL --upsample vae --checkpoint --task $task  --patch_count 2

fi