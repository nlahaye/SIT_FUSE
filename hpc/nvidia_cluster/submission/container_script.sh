#!/bin/bash

eval "$(conda shell.bash hook)"
export PYTHONPATH=${PYTHONPATH}:/home/mz23hu84/MultiModalObj-master:/home/mz23hu84/learnergy/
export LD_LIBRARY_PATH=/home/mz23hu84/.conda/agoodman/lib/:${CUDA_HOME}:${LD_LIBRARY_PATH}
conda activate agoodman
which conda
which python
which torchrun
#torchrun --nnodes=1 --nproc_per_node=2 dbn_learnergy.py --yaml ./config/dbn/misr_modis_fuse_dbn_demo_curiosity.yaml
#torchrun --nnodes=1 --nproc_per_node=2 dbn_learnergy.py --yaml ./config/dbn/misr_modis_fuse_dbn_demo_curiosity.yaml
#python3  dbn_learnergy.py --yaml ./config/dbn/misr_modis_fuse_dbn_demo_curiosity.yaml
  
torchrun --nnodes=1 --nproc_per_node=2 dbn_learnergy.py --yaml ./config/dbn/modis_volcano_dbn_curiosity.yaml

