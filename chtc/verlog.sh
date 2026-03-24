#!/bin/bash
source .env
pid=$1  # ranges from 0 to num_commands*num_jobs-1 
step=$2 # ranges from 0 to num_jobs-1
#cmd=`tr '*' ' ' <<< $3` # replace * with space
#echo $cmd

export HOME=$_CONDOR_SCRATCH_DIR
export TRANSFORMERS_CACHE=$_CONDOR_SCRATCH_DIR/models
export HF_DATASETS_CACHE=$_CONDOR_SCRATCH_DIR/datasets
export HF_MODULES_CACHE=$_CONDOR_SCRATCH_DIR/modules
export HF_METRICS_CACHE=$_CONDOR_SCRATCH_DIR/metrics
export HF_HOME=$_CONDOR_SCRATCH_DIR/hf_home

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # on CHTC machines, gpu names are *not* the usual 0-7 by default, so we rename them here
export TORCHINDUCTOR_CACHE_DIR=$_CONDOR_SCRATCH_DIR/torch_cache
export TORCH_COMPILE_CACHE=$_CONDOR_SCRATCH_DIR/torch_compile_cache
export XDG_CACHE_HOME=$_CONDOR_SCRATCH_DIR/xdg_cache
export XDG_CONFIG_HOME=$_CONDOR_SCRATCH_DIR/xdg_config
export _USAGE_STATS_JSON_PATH=$_CONDOR_SCRATCH_DIR/vllm_usage
export VLLM_USAGE_DISABLE=1

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_P2P_DISABLE=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export OUTLINES_CACHE_DIR='/tmp/.outlines'
export USER=${CHTC_USER}
export RAY_TMPDIR=/tmp/ray_$USER
export HYDRA_FULL_ERROR=1

# Transfer code from staging (Assumes you packed Verlog into Verlog.tar.gz)
cp /staging/${USER}/Verlog.tar.gz .
tar -xzf Verlog.tar.gz
rm Verlog.tar.gz

# --- Install Verlog into the container's Python ---
cd Verlog
pip install -e .
pip install packaging

export PYTHONPATH=.:$PYTHONPATH

# Ensure huggingface and wandb tokens if needed:
# (Make sure WANDB_API_KEY and HF_TOKEN are passed in your .sub file!)
wandb login ${WANDB_API_KEY}
# hf auth login --token ${HF_TOKEN}

# Run training (sized for 1 GPU with 3B model)
NUM_ENVS=8
BATCH_SIZE=64
MINI_BATCH_SIZE=$((BATCH_SIZE / 2))
MICRO_BATCH_SIZE=4
FORWARD_BATCH_SIZE=$((4 * MICRO_BATCH_SIZE))
OFFLOAD=true
PPO_EPOCHS=2

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='babyai_ppo' \
    algorithm.adv_estimator=gae \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_ENVS} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    trainer.balance_batch=False \
    trainer.critic_warmup=10 \
    trainer.critic_warmup_batch_repeat_times=40 \
    trainer.critic_warmup_batch_divide_ratio=4 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='time_aware_balrog' \
    trainer.experiment_name='babyai_ppo_test' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=30 \
    trainer.total_epochs=60 \
    trainer.val_before_train=True \
    envs.num_envs=${NUM_ENVS} \
    envs.env_name=babyai \
    envs.task=BabyAI-MixedTrainLocal-v0/goto \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-3B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_epochs=${PPO_EPOCHS} \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    critic.model.fsdp_config.param_offload=${OFFLOAD} \
    critic.model.fsdp_config.optimizer_offload=${OFFLOAD} \
    critic.ppo_max_token_len_per_gpu=8192 \
    critic.forward_max_token_len_per_gpu=8192 \
    critic.forward_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    data.train_files=$HOME/babyai/train.parquet \
    data.val_files=$HOME/babyai/test.parquet 2>&1 | tee verlog_run.log
