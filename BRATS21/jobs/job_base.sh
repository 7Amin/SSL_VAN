#!/bin/bash
#SBATCH --job-name=BRABase
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

if [ $job -eq 724 ]
then

    PYTHONPATH=. python3 BRATS21/main.py --squared_dice  --num_samples 1 --workers 8 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json --logdir ./runs/BraTS21_new/test_log --save_checkpoint  --max_epochs 500 \
  --distributed --use_normal_dataset --batch_size 1 --roi_x 128 --roi_y 128 --roi_z 128 --val_every 10 --model_inferer inferer \
  --valid_loader valid_loader --model_v SwinUNETR24 --checkpoint

elif [ $job -eq 736 ]
then

  PYTHONPATH=. python3 BRATS21/main.py --squared_dice  --num_samples 1 --workers 8 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json --logdir ./runs/BraTS21_new/test_log --save_checkpoint  --max_epochs 500 \
  --distributed --use_normal_dataset --batch_size 1 --roi_x 128 --roi_y 128 --roi_z 128 --val_every 10 --model_inferer inferer \
  --valid_loader valid_loader --model_v SwinUNETR36 --checkpoint

elif [ $job -eq 748 ]
then

  PYTHONPATH=. python3 BRATS21/main.py --squared_dice  --num_samples 1 --workers 8 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json --logdir ./runs/BraTS21_new/test_log --save_checkpoint  --max_epochs 500 \
  --distributed --use_normal_dataset --batch_size 1 --roi_x 128 --roi_y 128 --roi_z 128 --val_every 10 --model_inferer inferer \
  --valid_loader valid_loader --model_v SwinUNETR48 --checkpoint

elif [ $job -eq 816 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice  --num_samples 1 --workers 8 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json --logdir ./runs/BraTS21_new/test_log --save_checkpoint  --max_epochs 500 \
  --distributed --use_normal_dataset --batch_size 1 --roi_x 128 --roi_y 128 --roi_z 128 --val_every 10 --model_inferer inferer \
  --valid_loader valid_loader --model_v UNETR16 --checkpoint

elif [ $job -eq 832 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice  --num_samples 1 --workers 8 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json --logdir ./runs/BraTS21_new/test_log --save_checkpoint  --max_epochs 500 \
  --distributed --use_normal_dataset --batch_size 1 --roi_x 128 --roi_y 128 --roi_z 128 --val_every 10 --model_inferer inferer \
  --valid_loader valid_loader --model_v UNETR32 --checkpoint

elif [ $job -eq 900 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice  --num_samples 1 --workers 8 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json --logdir ./runs/BraTS21_new/test_log --save_checkpoint  --max_epochs 500 \
  --distributed --use_normal_dataset --batch_size 1 --roi_x 128 --roi_y 128 --roi_z 128 --val_every 10 --model_inferer inferer \
  --valid_loader valid_loader --model_v nnUNet --checkpoint

elif [ $job -eq 1000 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice  --num_samples 1 --workers 8 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json --logdir ./runs/BraTS21_new/test_log --save_checkpoint  --max_epochs 500 \
  --distributed --use_normal_dataset --batch_size 1 --roi_x 128 --roi_y 128 --roi_z 128 --val_every 10 --model_inferer inferer \
  --valid_loader valid_loader --model_v SegResNetVAE --checkpoint

elif [ $job -eq 1100 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice  --num_samples 1 --workers 8 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json --logdir ./runs/BraTS21_new/test_log --save_checkpoint  --max_epochs 500 \
  --distributed --use_normal_dataset --batch_size 1 --roi_x 128 --roi_y 128 --roi_z 128 --val_every 10 --model_inferer inferer \
  --valid_loader valid_loader --model_v AttentionUnet --checkpoint

elif [ $job -eq 1200 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice  --num_samples 1 --workers 8 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json --logdir ./runs/BraTS21_new/test_log --save_checkpoint  --max_epochs 500 \
  --distributed --use_normal_dataset --batch_size 1 --roi_x 128 --roi_y 128 --roi_z 128 --val_every 10 --model_inferer inferer \
  --valid_loader valid_loader --model_v Unetpp --checkpoint

elif [ $job -eq 1300 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice  --num_samples 1 --workers 8 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json --logdir ./runs/BraTS21_new/test_log --save_checkpoint  --max_epochs 500 \
  --distributed --use_normal_dataset --batch_size 1 --roi_x 128 --roi_y 128 --roi_z 128 --val_every 10 --model_inferer inferer \
  --valid_loader valid_loader --model_v BasicUNetPlusPlus --checkpoint

elif [ $job -eq 1410 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice  --num_samples 1 --workers 8 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json --logdir ./runs/BraTS21_new/test_log --save_checkpoint  --max_epochs 500 \
  --distributed --use_normal_dataset --batch_size 1 --roi_x 128 --roi_y 128 --roi_z 128 --val_every 10 --model_inferer inferer \
  --valid_loader valid_loader --model_v DiNTS_Search --checkpoint

elif [ $job -eq 1420 ]
then
  PYTHONPATH=. python3 BRATS21/main.py --squared_dice  --num_samples 1 --workers 8 --base_data ../images/BraTS21 \
  --json_list input_list/dataset_BRATS21_List.json --logdir ./runs/BraTS21_new/test_log --save_checkpoint  --max_epochs 500 \
  --distributed --use_normal_dataset --batch_size 1 --roi_x 128 --roi_y 128 --roi_z 128 --val_every 10 --model_inferer inferer \
  --valid_loader valid_loader --model_v DiNTS_Instance --checkpoint
fi