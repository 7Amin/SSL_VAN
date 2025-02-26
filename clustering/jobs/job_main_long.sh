#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH -p mi100-long
#SBATCH --gpus-per-node=0
#SBATCH -G 0
#SBATCH --mem=96gb


cd SSL_VAN
module spider cuda
conda activate ssl_van_seg

n_clusters_range=$(seq 80 500)

num_runs=400

for (( i=1; i<=$num_runs; i++ ))
do
    n_clusters=$(shuf -n 1 -e $n_clusters_range)

    PYTHONPATH=. python3 clustering/kmeans_learning.py \
             --num_workers 64 \
             --mode server \
             --num_samples 1 \
             --km_path ./cluster_models_1/cluster_model_1_{}_{}_{}.joblib \
             --n_clusters "$n_clusters" \
             --max_iter 350 \
             --batch_size 20 \
             --n_init 50 \
             --num_runs $num_runs

done
