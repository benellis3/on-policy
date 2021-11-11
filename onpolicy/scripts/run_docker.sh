#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
WANDB_API_KEY=$(cat $HOME/.netrc | grep 'password' | awk '{print $2}')
GPU=$1
name=${USER}_pymarl_GPU_${GPU}_${HASH}

echo "Launching container named '${name}' on GPU '${GPU}'"
# Launches a docker container using our image, and runs the provided command

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU="$GPU" ${cmd} run  \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_CONFIG_DIR=/home/mingfei/marl_ppo \
    --name $name \
    --user $(id -u) \
    --memory-reservation 30g \
    -v $(pwd):/home/mingfei/marl_ppo \
    -t mappo:ben_smac \
    ${@:2}
