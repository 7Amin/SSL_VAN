#!/bin/bash
#SBATCH --job-name=new_SSL_VAN
#SBATCH --time=05:59:59

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
#SBATCH -p a100

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export PYTHONPATH=.
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
conda activate ssl_van_seg

### the command to run

srun python3 BTCV/main.py --workers 8 --base_data ../images/BTCV/Abdomen/RawData/Training \
  --json_list input_list/dataset_BTCV_List.json --logdir ./runs/BTCV/test_log --save_checkpoint --max_epochs 5000 \
  --distributed --use_normal_dataset --batch_size 4 --num_stages 4 --embed_dims 128 128 512 512 --depths 3 4 4 4 \
  --mlp_ratios 4 4 4 4 --roi_x 96 --roi_y 96 --roi_z 96 --val_every 10 --model_inferer inferer \
  --valid_loader valid_loader --model_v VANV4121double --upsample vae --checkpoint