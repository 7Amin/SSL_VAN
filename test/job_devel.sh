#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --time=01:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH -p devel
#SBATCH --gpus-per-node=0
#SBATCH -G 0
#SBATCH --mem=96gb


cd SSL_VAN
module spider cuda
conda activate ssl_van_seg



PYTHONPATH=. python3 test/test_TCIA_COLON.py
