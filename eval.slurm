#!/bin/bash

# this is a multi-node SLURM script using `accelerate` launcher

#SBATCH --job-name=eval_llm
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per node
#SBATCH --gres=gpu:8                 # EDIT this if it's not 8-gpus per node
#SBATCH --exclusive
#SBATCH --output=/mnt/home/f08944064/lighteval/logs/%x-%j.out
#SBATCH --error=/mnt/home/f08944064/lighteval/logs/%x-%j.err

echo "START TIME: $(date)"

source ~/.bashrc
source activate lighteval

# auto-fail on any errors in this script
set -eo pipefail

# logging script's variables/commands for future debug needs
set -x

# EDIT the conda evn and any startup scripts
# source /path/to/start-xxx-user # if you have something to preload before the job
# conda activate stas-xxx        # if you have conda env to activate

LOG_PATH="main_log.txt"

# EDIT if it's not 8-gpus per node
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# define the node 0 hostname:port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# note `\$SLURM_PROCID` we don't want it interpolated till `srun` since otherwise all nodes will get
# 0 and the launcher will hang
#
# same goes for `\$(hostname -s|tr -dc '0-9')` - we want it to interpolate at `srun` time
LAUNCHER="python -u -m accelerate.commands.launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_machines $NNODES \
    --num_processes 1 \
    --role \$(hostname -s|tr -dc '0-9'): --tee 3 \
"

# EDIT the path+name of the python script and whatever args it needs
export PROGRAM="\
    run_evals_accelerate.py \
    --model_args "pretrained=${1},trust_remote_code=True" \
    --override_batch_size 1 \
    --output_dir="./evals/" \
    --tasks tasks_examples/open_llm_leaderboard_tasks.txt \
    --model_parallel \
    --use_chat_template \
"
    #--tasks tasks_examples/twllm_eval.txt \
    #--custom_tasks "community_tasks/twllm_eval.py" \

export CMD="$LAUNCHER $PROGRAM"

echo $CMD

# EDIT if you want to redirect /tmp to /scratch (some local SSD path) since /tmp is tiny on compute nodes
# export TMPDIR=/scratch

# EDIT: useful for debug if needed
#
# to debug NCCL issues
# export NCCL_DEBUG=INFO
#
# to unravel async errors w/o the correct traceback - potentially makes everything very slower
# export CUDA_LAUNCH_BLOCKING=1
#
# to force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    "

# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"
