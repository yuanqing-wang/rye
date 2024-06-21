#!/bin/bash
#SBATCH --time=47:59:00
#SBATCH --mem=20GB
#SBATCH --job-name=qm9
#SBATCH --output=%A.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

python test_mnist.py

