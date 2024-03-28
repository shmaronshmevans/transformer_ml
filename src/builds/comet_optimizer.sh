#!/bin/bash

#SBATCH --job-name=com_opt
#SBATCH --account=ai2es
#SBATCH --qos=long
#SBATCH --time=08:00:00
#SBATCH --partition=v100
#SBATCH --cpus-per-task=16
#SBATCH --mem=300gb
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Number of processes
#SBATCH --error=/home/aevans/slurm_error_logs/ai2es_error_%j.err
#SBATCH --output=/home/aevans/slurm_output_logs/ai2es_log_%j.out

export APPTAINER_BIND="/rdma/dgx-a100/NYSM/:/home/aevans/nysm, /rdma/xcitedb/AI2ES/:/home/aevans/ai2es"
export CUDA_VISIBLE_DEVICES="0"
export COMET_API KEY="leAiWyR5Ck7tkdiHIT7n6QWNa"

apptainer run --nv /home/aevans/apptainer/pytorch.sif /bin/bash -c 'conda activate base; /home/aevans/miniconda3/bin/comet optimize -j 2 /home/aevans/transformer_ml/src/comet_hyperparams_opt.py /home/aevans/transformer_ml/src/optimizer.config' 