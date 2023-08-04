
#!/bin/bash

eval "$(conda shell.bash hook)"
export PYTHONPATH=${PYTHONPATH}:$WORK/STI_FUSE/:$WORK/learnergy/
export LD_LIBRARY_PATH=${CUDA_HOME}:${LD_LIBRARY_PATH}

torchrun --nnodes=1 --nproc_per_node=1 dbn_learnergy.py --yaml ./config/dbn/modis_volcano_dbn_curiosity.yaml




