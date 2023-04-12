#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --time=23:59:59
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH -p a100
#SBATCH --gpus-per-node=2
#SBATCH -G 8
#SBATCH --mem=80gb


n_clusters_range=$(seq 50 500)

# Define the number of times to run the command
num_runs=400

# Loop over the number of runs
for (( i=1; i<=$num_runs; i++ ))
do
    # Select a random value for n_clusters
    n_clusters=$(shuf -n 1 -e $n_clusters_range)

    # Construct the command with the selected value for n_clusters
    command="cd SSL_VAN && \
             module spider cuda && \
             conda activate ssl_van_seg && \
             PYTHONPATH=. python3 clustering/kmeanis_learning.py \
             --num_workers 8 \
             --mode server \
             --num_samples 6 \
             --km_path ./cluster_models/cluster_model_${n_clusters}_${i}.joblib \
             --n_clusters $n_clusters \
             --max_iter 100 \
             --batch_size 20 \
             --n_init 50"

    # Run the command
    $command
done
