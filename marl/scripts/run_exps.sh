#!/bin/bash
# pipeline for running sets of experiments on linux machine 
# references: 
# https://www.slashroot.in/how-run-multiple-commands-parallel-linux
# https://linuxhandbook.com/run-process-background/
# https://askubuntu.com/questions/674333/how-to-pass-an-array-as-function-argument
# https://stackoverflow.com/questions/394230/how-to-detect-the-os-from-a-bash-script


# # enable job control  
# set -m
####################################################################
CUR_DIR=`pwd`

# options 
EXP_SET="rnn"
DISCRETE_FLAG="true"
ACTION_TYPE="discrete"


POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--exp)
    EXP_SET="$2"
    shift # past argument
    shift # past value
    ;;
    -a|--act)
    ACTION_TYPE="$2"
    if [ "$ACTION_TYPE" == "discrete" ]; then
        DISCRETE_FLAG="true"
    else
        DISCRETE_FLAG="false"
    fi
    shift # past argument
    shift # past value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


# run all commands in given array concurrently
# either spawn new terminal to run or run in background
run_concurrent () {
    arr=("$@")
    for i in "${arr[@]}";
        do
            if [[ "$OSTYPE" == "linux-gnu" ]]; then
                cd $CUR_DIR; source activate interact; eval $i &
            elif [[ "$OSTYPE" == "darwin"* ]]; then
                osascript -e "tell app \"Terminal\" to do script \"cd $CUR_DIR; source activate interact; $i\""
            else
                echo "Unknown os, cannot execute $i"
            fi
        done
}

####################################################################
# Sanity check on goal spread with RNN
benchmark_goal_spread () {
    run1="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir rnn_benchmark  --env_config mpe_hierarchy/configs/goal_spread.yaml --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 10 --tag mlp" 

    run2="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir rnn_benchmark  --env_config mpe_hierarchy/configs/goal_spread.yaml --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 10 --actor rnn --tag rnn-actor"

    run3="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir rnn_benchmark  --env_config mpe_hierarchy/configs/goal_spread.yaml --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 10 --critic rnn --tag rnn-critic"

    run4="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir rnn_benchmark  --env_config mpe_hierarchy/configs/goal_spread.yaml --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 10 --actor rnn --critic rnn --tag rnn-ac"

    # run all in parallel
    run_concurrent "$run1" "$run2" "$run3" "$run4" 
}

####################################################################
# RNN policy comparisons 
benchmark_rnn_policy () {
    run1="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir rnn_benchmark  --env_config mpe_hierarchy/configs/simple_spread.yaml --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 10 --tag mlp" 

    run2="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir rnn_benchmark  --env_config mpe_hierarchy/configs/simple_spread.yaml --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 10 --actor rnn --tag rnn-actor"

    run3="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir rnn_benchmark  --env_config mpe_hierarchy/configs/simple_spread.yaml --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 10 --critic rnn --tag rnn-critic"

    run4="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir rnn_benchmark  --env_config mpe_hierarchy/configs/simple_spread.yaml --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 10 --actor rnn --critic rnn --tag rnn-ac"

    # run all in parallel
    run_concurrent "$run1" "$run2" "$run3" "$run4" 
}

####################################################################
# Simple communication 
run_discrete_comm_ablation () {
    run1="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir debug --scenario simple_speaker_listener --env_config mpe_hierarchy/configs/simple_speaker_listener.yaml --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 10 --overwrite discrete_action_space-bool-${DISCRETE_FLAG} --tag sl-${ACTION_TYPE}-normal"

    run2="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir debug --scenario simple_speaker_listener --env_config mpe_hierarchy/configs/simple_speaker_listener.yaml --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 10 --overwrite discrete_action_space-bool-${DISCRETE_FLAG} use_oracle_dist-bool-true --tag sl-${ACTION_TYPE}-oracle-dist"

    run3="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir debug --scenario simple_speaker_listener --env_config mpe_hierarchy/configs/simple_speaker_listener.yaml --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 10 --overwrite discrete_action_space-bool-${DISCRETE_FLAG} use_oracle_pos-bool-true --tag sl-${ACTION_TYPE}-oracle-pos"

    run4="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir debug --scenario simple_speaker_listener --env_config mpe_hierarchy/configs/simple_speaker_listener.yaml --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 10 --overwrite discrete_action_space-bool-${DISCRETE_FLAG} use_oracle_speaker-bool-true --tag sl-${ACTION_TYPE}-oracle-speaker"

    run5="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir debug --scenario simple_speaker_listener --env_config mpe_hierarchy/configs/simple_speaker_listener.yaml --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 10 --overwrite discrete_action_space-bool-${DISCRETE_FLAG} use_oracle_speaker_goal-bool-true --tag sl-${ACTION_TYPE}-oracle-speaker-goal"

    # run all in parallel
    run_concurrent "$run1" "$run2" "$run3" "$run4" "$run5"
}

####################################################################
# With partial observable tasks 
run_simple_tag () {
    run1="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir test  --env_config mpe_hierarchy/configs/simple_tag.yaml --scenario simple_tag --use_tensorboard --n_episodes 600000 --seed 3 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 5 --tag test" 

    run2="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir test  --env_config mpe_hierarchy/configs/simple_tag.yaml --scenario simple_tag --use_tensorboard --n_episodes 600000 --seed 4 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 5 --tag test"

    run3="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir test  --env_config mpe_hierarchy/configs/simple_tag.yaml --scenario simple_tag --use_tensorboard --n_episodes 600000 --seed 5 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 5 --tag test"

    run4="python algorithms/rmaddpg/run_rmaddpg.py --sub_dir test  --env_config mpe_hierarchy/configs/simple_tag.yaml --scenario simple_tag --use_tensorboard --n_episodes 600000 --seed 6 --n_rollout_threads 2 --sample_batch_size 4 --n_updates_per_train 5 --tag test"

    # run all in parallel
    run_concurrent "$run1" "$run2" "$run3" "$run4" 
}


####################################################################
# With partial observable tasks 
run_partial_tasks () {
    echo "partial"
}


####################################################################

# run experiments 
if [ "$EXP_SET" == "rnn" ]
then
    benchmark_rnn_policy
elif [ "$EXP_SET" == "goal" ]
then
    benchmark_goal_spread
elif [ "$EXP_SET" == "comm" ]
then
    run_discrete_comm_ablation
elif [ "$EXP_SET" == "comm" ]
then
    run_simple_tag
elif [ "$EXP_SET" == "partial" ]
then
    run_partial_tasks
else
    echo "Nothing to run..."
fi

