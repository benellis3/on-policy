#!/bin/bash

env="StarCraft2"
gpu=$1
map=$2
exp=$3


get_lr(){
    grep $map $(pwd)/parameters.txt | awk '{print $2}'
}

get_activation(){
    grep $map $(pwd)/parameters.txt | awk '{print $3}'
}

get_epochs(){
    grep $map $(pwd)/parameters.txt | awk '{print $4}'
}

get_minibatch(){
    grep $map $(pwd)/parameters.txt | awk '{print $5}'
}

get_clip(){
    grep $map $(pwd)/parameters.txt | awk '{print $6}'
}

get_gain(){
    grep $map $(pwd)/parameters.txt | awk '{print $7}'
}

get_network(){
    grep $map $(pwd)/parameters.txt | awk '{print $8}'
}

get_stacked_frames(){
    grep $map $(pwd)/parameters.txt | awk '{print $9}'
}

ARGS="--lr $(get_lr) --ppo_epoch $(get_epochs) --num_mini_batch $(get_minibatch)
--gain $(get_gain) --clip_param $(get_clip)"

# boolean arguments. Specifying these does the opposite of what the argument sounds like
# (store false), which is absolutely mental but there you go.
if [[ ${get_activation} != "ReLU" ]]; then
    ARGS="$ARGS --use_ReLU"
fi

if [[ ${get_network} != "rnn" ]]; then
    ARGS="$ARGS --use_recurrent_policy"
    algo="mappo"
else
    algo="rmappo"
fi


if [ ${get_stacked_frames} > 1 ]; then
    ARGS="$ARGS --use_stacked_frames --stacked_frames $(get_stacked_frames)"
fi

./onpolicy/scripts/run_docker.sh $gpu python -m onpolicy.scripts.train.train_smac --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
    --n_training_threads 127 --n_rollout_threads 8 --episode_length 400 \
    --num_env_steps 10000000 --use_value_active_masks --use_eval \
    ${ARGS}
