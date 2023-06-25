#!/bin/bash
#SBATCH --job-name=BTCVvae
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

if [ $job -eq 47 ]
then
  PYTHONPATH=. python3 BTCV/main.py --use_ssl_pretrained --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json  --logdir ./runs/BTCV_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4 --upsample vae --checkpoint \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV4_20_500_256_True_5__final.pt

elif [ $job -eq 48 ]
then
  PYTHONPATH=. python3 BTCV/main.py --use_ssl_pretrained --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json  --logdir ./runs/BTCV_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4 --upsample vae --checkpoint \
  --pretrained_model_name pre_train_96-192-384-768_3-3-24-3_8-8-4-4_vae_inferer_valid_loader_PREVANV4_20_500_256_True_5__final.pt

elif [ $job -eq 427 ]
then
  PYTHONPATH=. python3 BTCV/main.py --use_ssl_pretrained --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json  --logdir ./runs/BTCV_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint --patch_count 2 \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV4GL_2_20_500_256_True_5__final.pt

elif [ $job -eq 428 ]
then
  PYTHONPATH=. python3 BTCV/main.py --use_ssl_pretrained --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json  --logdir ./runs/BTCV_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4GL --upsample vae --checkpoint --patch_count 2 \
  --pretrained_model_name pre_train_96-192-384-768_3-3-24-3_8-8-4-4_vae_inferer_valid_loader_PREVANV4GL_2_20_500_256_True_5__final.pt

elif [ $job -eq 527 ]
then
  PYTHONPATH=. python3 BTCV/main.py --use_ssl_pretrained --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json  --logdir ./runs/BTCV_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV5GL --upsample vae --checkpoint --patch_count 2 \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV5GL_2_20_500_256_True_5__final.pt

elif [ $job -eq 528 ]
then
  PYTHONPATH=. python3 BTCV/main.py --use_ssl_pretrained --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json  --logdir ./runs/BTCV_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV5GL --upsample vae --checkpoint --patch_count 2 \
  --pretrained_model_name pre_train_96-192-384-768_3-3-24-3_8-8-4-4_vae_inferer_valid_loader_PREVANV5GL_2_20_500_256_True_5__final.pt

elif [ $job -eq 627 ]
then
  PYTHONPATH=. python3 BTCV/main.py --use_ssl_pretrained --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json  --logdir ./runs/BTCV_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 2 --num_stages 4 --embed_dims 64 128 256 512 --depths 3 4 6 3 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV6GL --upsample vae --checkpoint --patch_count 2 \
  --pretrained_model_name pre_train_64-128-256-512_3-4-6-3_8-8-4-4_vae_inferer_valid_loader_PREVANV6GL_2_20_500_256_True_5__final.pt

elif [ $job -eq 628 ]
then
  PYTHONPATH=. python3 BTCV/main.py --use_ssl_pretrained --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json  --logdir ./runs/BTCV_new/test_log --save_checkpoint --max_epochs 15000 \
  --distributed --use_normal_dataset --batch_size 1 --num_stages 4 --embed_dims 96 192 384 768 --depths 6 6 90 6 \
  --mlp_ratios 8 8 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 1 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV6GL --upsample vae --checkpoint --patch_count 2 \
  --pretrained_model_name pre_train_96-192-384-768_3-3-24-3_8-8-4-4_vae_inferer_valid_loader_PREVANV6GL_2_20_500_256_True_5__final.pt

fi