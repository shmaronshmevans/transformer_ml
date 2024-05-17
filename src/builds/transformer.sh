#!/bin/bash

#SBATCH --job-name=__ViT
#SBATCH --account=ai2es
#SBATCH --qos=long
#SBATCH --time=96:00:00
#SBATCH --partition=a6000
#SBATCH --cpus-per-task=20
#SBATCH --mem=300G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Number of processes
#SBATCH --error=/home/aevans/slurm_error_logs/ai2es_error_%j.err
#SBATCH --output=/home/aevans/slurm_output_logs/ai2es_log_%j.out

apptainer run --nv /home/aevans/apptainer/pytorch2.sif /home/aevans/miniconda3/bin/python /home/aevans/transformer_ml/src/engine.py

