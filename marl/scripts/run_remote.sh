#!/bin/bash
# example usage 
# fetch: bash scripts/run_remote.sh --mode fetch 
# --source ray_results/EXP_partial_spread_comm_large

# defaults 
MODE=""
GPU=0
CPU=8
MEM="15G"
SOURCE="exps"   # use * or .
TARGET="cluster_exps"
REMOTE_IP="justiny@q.vectorinstitute.ai"
REMOTE_DIR="/scratch/gobi1/justiny/marl/learn-to-interact/marl"


# parse arguments (only expose useful ones)
while (( "$#" )); do
  case "$1" in
    -m|--mode)
      MODE=$2
      shift 2
      ;;
    -g|--gpu)
      GPU=$2
      shift 2
      ;;
    -c|--cpu)
      CPU=$2
      shift 2
      ;;
    -s|--source)
      SOURCE=$2
      shift 2
      ;;
    -t|--target)
      TARGET=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done


# branch out 
if [ "$MODE" == "run" ]; then 
    # PARAMS should be job command, e.g. "python xx.py -option val ..."
    srun --gres=gpu:$GPU -c $CPU --mem=$MEM $PARAMS

elif [ "$MODE" == "debug" ]; then 
    srun --gres=gpu:$GPU -c $CPU --mem=$MEM -p interactive --pty $PARAMS

elif [ "$MODE" == "fetch" ]; then 
    # fetch result from remote to local
    scp -r $REMOTE_IP:$REMOTE_DIR/$SOURCE $TARGET

elif [ "$MODE" == "upload" ]; then 
    # NOTE: execute from local
    scp -r $SOURCE $REMOTE_IP:$REMOTE_DIR/$TARGET

elif [ "$MODE" == "port-forward" ]; then 
    # NOTE: execute from local
    # reference: https://www.ssh.com/ssh/tunneling/example
    ssh -L 16006:127.0.0.1:6006 $REMOTE_IP

else 
    echo "option not supported..."
fi 


# srun -p cpu --gres=gpu:0 -c 4 --mem=8G "<command>"
# srun -p cpu --gres=gpu:0 -c 8 --mem=15G "<command>"
# srun -p cpu --gres=gpu:0 -c 16 --mem=32G "<command>"