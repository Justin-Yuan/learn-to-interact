#!/bin/bash

# assumed in maddpg repo 
CUR_DIR=$(pwd)

# add mpe_hierarchy to PYTHONPATH
export $CUR_DIR/mpe_hierarchy:$PYTHONPATH

# # install dependencies
# cat requirements.txt | xargs -n 1 pip install
