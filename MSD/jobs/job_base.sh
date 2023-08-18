#!/bin/bash
#SBATCH --job-name=MSDBase
#SBATCH --time=05:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -p a100
#SBATCH --gpus-per-node=2
#SBATCH -G 2
#SBATCH --mem=100gb

cd SSL_VAN
module spider cuda
conda activate ssl_van_seg
#pip3 install -r ./requirements.txt

job=$1
task=$2
epochs=$3


if [ $job -eq 724 ]
then
  PYTHONPATH=. python3 MSD/main.py  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs $epochs \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v SwinUNETR24 --upsample vae --checkpoint --task $task

elif [ $job -eq 736 ]
then
  PYTHONPATH=. python3 MSD/main.py  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs $epochs \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v SwinUNETR36 --upsample vae --checkpoint --task $task

elif [ $job -eq 748 ]
then

  PYTHONPATH=. python3 MSD/main.py  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs $epochs \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v SwinUNETR48 --upsample vae --checkpoint --task $task --num_samples 2

elif [ $job -eq 816 ]
then
    PYTHONPATH=. python3 MSD/main.py  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs $epochs \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v UNETR16 --upsample vae --checkpoint --task $task

elif [ $job -eq 832 ]
then
    PYTHONPATH=. python3 MSD/main.py  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs $epochs \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v UNETR32 --upsample vae --checkpoint --task $task

elif [ $job -eq 900 ]
then
  PYTHONPATH=. python3 MSD/main.py  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs $epochs \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v nnUNet --upsample vae --checkpoint --task $task

elif [ $job -eq 1000 ]
then
  PYTHONPATH=. python3 MSD/main.py  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs $epochs \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v SegResNetVAE --upsample vae --checkpoint --task $task

elif [ $job -eq 1100 ]
then
    PYTHONPATH=. python3 MSD/main.py  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs $epochs \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v AttentionUnet --upsample vae --checkpoint --task $task

elif [ $job -eq 1200 ]
then
  PYTHONPATH=. python3 MSD/main.py  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs $epochs \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v Unetpp --upsample vae --checkpoint --task $task

elif [ $job -eq 1300 ]
then
  PYTHONPATH=. python3 MSD/main.py  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs $epochs \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v BasicUNetPlusPlus --upsample vae --checkpoint --task $task

elif [ $job -eq 1410 ]
then
  PYTHONPATH=. python3 MSD/main.py  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs $epochs \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v DiNTS_Search --upsample vae --checkpoint --task $task

elif [ $job -eq 1420 ]
then
  PYTHONPATH=. python3 MSD/main.py  --workers 8 --base_data ../images/MSD \
  --json_list input_list/dataset_MSD_List.json --logdir ./runs/MSD_new/test_log --save_checkpoint --max_epochs $epochs \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v DiNTS_Instance --upsample vae --checkpoint --task $task
fi