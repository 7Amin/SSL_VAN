#!/bin/bash
#SBATCH --job-name=BTCVvae
#SBATCH --time=01:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -p devel
#SBATCH --gpus-per-node=2
#SBATCH -G 2
#SBATCH --mem=80gb

cd SSL_VAN
module spider cuda
conda activate ssl_van_seg
#pip3 install -r ./requirements.txt


job=$1

if [ $job -eq 1 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --upsample vae  --checkpoint

elif [ $job -eq 2 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --upsample vae --checkpoint

elif [ $job -eq 3 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --upsample vae --checkpoint


elif [ $job -eq 4 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --upsample vae \
   --checkpoint

elif [ $job -eq 5 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --upsample vae \
  --checkpoint

elif [ $job -eq 6 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --upsample vae \
  --checkpoint


elif [ $job -eq 7 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --upsample vae --checkpoint

elif [ $job -eq 8 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --upsample vae --checkpoint

elif [ $job -eq 9 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --upsample vae --checkpoint

elif [ $job -eq 21 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV2 --upsample vae --checkpoint

elif [ $job -eq 22 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV2 --upsample vae --checkpoint

elif [ $job -eq 23 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1  --model_v VANV2 --upsample vae --checkpoint


elif [ $job -eq 24 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV2 \
  --upsample vae --checkpoint

elif [ $job -eq 25 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV2 \
  --upsample vae --checkpoint

elif [ $job -eq 26 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV2 \
  --upsample vae --checkpoint


elif [ $job -eq 27 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV2 --upsample vae --checkpoint

elif [ $job -eq 28 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV2 --upsample vae --checkpoint

elif [ $job -eq 29 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV2 --upsample vae --checkpoint



elif [ $job -eq 31 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV3 --upsample vae --checkpoint

elif [ $job -eq 32 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV3 --upsample vae --checkpoint

elif [ $job -eq 33 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1  --model_v VANV3 --upsample vae --checkpoint


elif [ $job -eq 34 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV3 \
  --upsample vae --checkpoint

elif [ $job -eq 35 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV3 \
  --upsample vae --checkpoint

elif [ $job -eq 36 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV3 \
  --upsample vae --checkpoint


elif [ $job -eq 37 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV3 --upsample vae --checkpoint

elif [ $job -eq 38 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV3 --upsample vae --checkpoint

elif [ $job -eq 39 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV3 --upsample vae --checkpoint


elif [ $job -eq 41 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4 --upsample vae --checkpoint

elif [ $job -eq 42 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4 --upsample vae --checkpoint

elif [ $job -eq 43 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1  --model_v VANV4 --upsample vae --checkpoint


elif [ $job -eq 44 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4 \
  --upsample vae --checkpoint

elif [ $job -eq 45 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4 \
  --upsample vae --checkpoint

elif [ $job -eq 46 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4 \
  --upsample vae --checkpoint


elif [ $job -eq 47 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4 --upsample vae --checkpoint

elif [ $job -eq 48 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4 --upsample vae --checkpoint

elif [ $job -eq 49 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4 --upsample vae --checkpoint

elif [ $job -eq 411 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4GL --upsample vae --checkpoint

elif [ $job -eq 412 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_v VANV4GL --upsample vae --checkpoint

elif [ $job -eq 413 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1  --model_v VANV4GL --upsample vae --checkpoint


elif [ $job -eq 414 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4GL \
  --upsample vae --checkpoint

elif [ $job -eq 415 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4GL \
  --upsample vae --checkpoint

elif [ $job -eq 416 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer --model_v VANV4GL \
  --upsample vae --checkpoint


elif [ $job -eq 417 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint

elif [ $job -eq 418 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint

elif [ $job -eq 419 ]
then
  PYTHONPATH=. python3 BTCV/main.py  --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 96 192 384 768 --depths 3 3 24 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint
fi


