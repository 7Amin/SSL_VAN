#!/bin/bash
#SBATCH --job-name=MSDvae
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

if [ $job -eq 1 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --upsample vae  --checkpoint --task $task

elif [ $job -eq 2 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --upsample vae --checkpoint --task $task

elif [ $job -eq 3 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --upsample vae --checkpoint --task $task


elif [ $job -eq 4 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --upsample vae \
   --checkpoint --task $task

elif [ $job -eq 5 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --upsample vae \
  --checkpoint --task $task

elif [ $job -eq 6 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --upsample vae \
  --checkpoint --task $task


elif [ $job -eq 7 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --upsample vae --checkpoint --task $task

elif [ $job -eq 8 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --upsample vae --checkpoint --task $task

elif [ $job -eq 9 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --upsample vae --checkpoint --task $task

elif [ $job -eq 21 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV2 --upsample vae --checkpoint --task $task

elif [ $job -eq 22 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV2 --upsample vae --checkpoint --task $task

elif [ $job -eq 23 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1  --model_v VANV2 --upsample vae --checkpoint --task $task


elif [ $job -eq 24 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV2 \
  --upsample vae --checkpoint --task $task

elif [ $job -eq 25 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV2 \
  --upsample vae --checkpoint --task $task

elif [ $job -eq 26 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV2 \
  --upsample vae --checkpoint --task $task


elif [ $job -eq 27 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV2 --upsample vae --checkpoint --task $task

elif [ $job -eq 28 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV2 --upsample vae --checkpoint --task $task

elif [ $job -eq 29 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV2 --upsample vae --checkpoint --task $task



elif [ $job -eq 31 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV3 --upsample vae --checkpoint --task $task

elif [ $job -eq 32 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV3 --upsample vae --checkpoint --task $task

elif [ $job -eq 33 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1  --model_v VANV3 --upsample vae --checkpoint --task $task


elif [ $job -eq 34 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV3 \
  --upsample vae --checkpoint --task $task

elif [ $job -eq 35 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV3 \
  --upsample vae --checkpoint --task $task

elif [ $job -eq 36 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV3 \
  --upsample vae --checkpoint --task $task


elif [ $job -eq 37 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV3 --upsample vae --checkpoint --task $task

elif [ $job -eq 38 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV3 --upsample vae --checkpoint --task $task

elif [ $job -eq 39 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV3 --upsample vae --checkpoint --task $task


elif [ $job -eq 41 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4 --upsample vae --checkpoint --task $task

elif [ $job -eq 42 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4 --upsample vae --checkpoint --task $task

elif [ $job -eq 43 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1  --model_v VANV4 --upsample vae --checkpoint --task $task


elif [ $job -eq 44 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4 \
  --upsample vae --checkpoint --task $task

elif [ $job -eq 45 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4 \
  --upsample vae --checkpoint --task $task

elif [ $job -eq 46 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4 \
  --upsample vae --checkpoint --task $task


elif [ $job -eq 47 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4 --upsample vae --checkpoint --task $task

elif [ $job -eq 48 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4 --upsample vae --checkpoint --task $task

elif [ $job -eq 49 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4 --upsample vae --checkpoint --task $task

elif [ $job -eq 421 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4GL --upsample vae --checkpoint --task $task  \
  --patch_count 2

elif [ $job -eq 422 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4GL --upsample vae --checkpoint --task $task  \
  --patch_count 2

elif [ $job -eq 423 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1  --model_v VANV4GL --upsample vae --checkpoint --task $task  \
  --patch_count 2


elif [ $job -eq 424 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4GL \
  --upsample vae --checkpoint --task $task  --patch_count 2

elif [ $job -eq 425 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4GL \
  --upsample vae --checkpoint --task $task  --patch_count 2

elif [ $job -eq 426 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4GL \
  --upsample vae --checkpoint --task $task  --patch_count 2


elif [ $job -eq 427 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint --task $task  --patch_count 2

elif [ $job -eq 428 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint --task $task  --patch_count 2

elif [ $job -eq 429 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint --task $task  --patch_count 2


elif [ $job -eq 431 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4GL --upsample vae --checkpoint --task $task  \
  --patch_count 3

elif [ $job -eq 432 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4GL --upsample vae --checkpoint --task $task  \
  --patch_count 3

elif [ $job -eq 433 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1  --model_v VANV4GL --upsample vae --checkpoint --task $task  \
  --patch_count 3


elif [ $job -eq 434 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4GL \
  --upsample vae --checkpoint --task $task  --patch_count 3

elif [ $job -eq 435 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4GL \
  --upsample vae --checkpoint --task $task  --patch_count 3

elif [ $job -eq 436 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4GL \
  --upsample vae --checkpoint --task $task  --patch_count 3


elif [ $job -eq 437 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint --task $task  --patch_count 3

elif [ $job -eq 438 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint --task $task  --patch_count 3

elif [ $job -eq 439 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint --task $task  --patch_count 3


elif [ $job -eq 441 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4GL --upsample vae --checkpoint --task $task  \
  --patch_count 4

elif [ $job -eq 442 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4GL --upsample vae --checkpoint --task $task  \
  --patch_count 4

elif [ $job -eq 443 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1  --model_v VANV4GL --upsample vae --checkpoint --task $task  \
  --patch_count 4


elif [ $job -eq 444 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4GL \
  --upsample vae --checkpoint --task $task  --patch_count 4

elif [ $job -eq 445 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4GL \
  --upsample vae --checkpoint --task $task  --patch_count 4

elif [ $job -eq 446 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4GL \
  --upsample vae --checkpoint --task $task  --patch_count 4


elif [ $job -eq 447 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint --task $task  --patch_count 4

elif [ $job -eq 448 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint --task $task  --patch_count 4

elif [ $job -eq 449 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint --task $task  --patch_count 4

elif [ $job -eq 4121 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 4 3 \
  --mlp_ratios 6 6 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4GLV1 --upsample vae --checkpoint --task $task  \
  --patch_count 2

elif [ $job -eq 4221 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 3 --embed_dims 64 128 256  --depths 3 4 3 \
  --mlp_ratios 6 6 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4GLV2 --upsample vae --checkpoint --task $task  \
  --patch_count 2

elif [ $job -eq 4231 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 2 --embed_dims 128 256 --depths 3 4 \
  --mlp_ratios 6 6 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4GLV2 --upsample vae --checkpoint --task $task  \
  --patch_count 3

elif [ $job -eq 4241 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 2 --embed_dims 128 256 --depths 3 4 \
  --mlp_ratios 6 6 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4GLV2 --upsample vae --checkpoint --task $task  \
  --patch_count 4

elif [ $job -eq 521 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV5GL --upsample vae --checkpoint --task $task  \
  --patch_count 2

elif [ $job -eq 522 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV5GL --upsample vae --checkpoint --task $task  \
  --patch_count 2

elif [ $job -eq 523 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1  --model_v VANV5GL --upsample vae --checkpoint --task $task  \
  --patch_count 2


elif [ $job -eq 524 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV5GL \
  --upsample vae --checkpoint --task $task  --patch_count 2

elif [ $job -eq 525 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV5GL \
  --upsample vae --checkpoint --task $task  --patch_count 2

elif [ $job -eq 526 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV5GL \
  --upsample vae --checkpoint --task $task  --patch_count 2


elif [ $job -eq 527 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV5GL --upsample vae --checkpoint --task $task  --patch_count 2

elif [ $job -eq 528 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV5GL --upsample vae --checkpoint --task $task  --patch_count 2

elif [ $job -eq 529 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV5GL --upsample vae --checkpoint --task $task  --patch_count 2


elif [ $job -eq 621 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV6GL --upsample vae --checkpoint --task $task  \
  --patch_count 2

elif [ $job -eq 622 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV6GL --upsample vae --checkpoint --task $task  \
  --patch_count 2

elif [ $job -eq 623 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1  --model_v VANV6GL --upsample vae --checkpoint --task $task  \
  --patch_count 2


elif [ $job -eq 624 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV6GL \
  --upsample vae --checkpoint --task $task  --patch_count 2

elif [ $job -eq 625 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV6GL \
  --upsample vae --checkpoint --task $task  --patch_count 2

elif [ $job -eq 626 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV6GL \
  --upsample vae --checkpoint --task $task  --patch_count 2


elif [ $job -eq 627 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV6GL --upsample vae --checkpoint --task $task  --patch_count 2

elif [ $job -eq 628 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV6GL --upsample vae --checkpoint --task $task  --patch_count 2

elif [ $job -eq 629 ]
then
  PYTHONPATH=. python3 MSD/main.py  --test_mode  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV6GL --upsample vae --checkpoint --task $task  --patch_count 2
fi