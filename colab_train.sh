#!/bin/bash
set -e

echo "============================================"
echo "1. Installing Dependencies..."
echo "============================================"
pip install packaging gymnasium minigrid pandas pyarrow
pip install torch==2.3.0
pip install vllm==0.5.4 ray

# Install verl
if [ ! -d "verl" ]; then
    git clone https://github.com/verl-project/verl.git -b v0.7.0
fi
# Strip flash-attn from verl's requirements so it doesn't try to build it from source
sed -i '/flash-attn/d' verl/setup.py
sed -i '/flash-attn/d' verl/requirements.txt 2>/dev/null || true
pip install -e verl
# Install Verlog
pip install -e .

echo "============================================"
echo "2. Generating BabyAI Data..."
echo "============================================"
# This creates ~/data/babyai/train.parquet and test.parquet
python3 chtc/generate_babyai_data.py

echo "============================================"
echo "3. Starting PPO Training..."
echo "============================================"
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export NCCL_P2P_DISABLE=1

# Sized for 1 GPU (A100 40GB or similar in Colab Pro)
NUM_ENVS=2
BATCH_SIZE=64
MINI_BATCH_SIZE=$((BATCH_SIZE / 2))
MICRO_BATCH_SIZE=4
FORWARD_BATCH_SIZE=$((4 * MICRO_BATCH_SIZE))
OFFLOAD=false
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
    trainer.experiment_name='babyai_ppo_test_colab' \
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
    data.train_files=$HOME/data/babyai/train.parquet \
    data.val_files=$HOME/data/babyai/test.parquet 2>&1 | tee colab_run.log
