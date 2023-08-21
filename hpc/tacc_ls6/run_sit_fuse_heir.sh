
#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:/app/rundir/SIT_FUSE/:/app/rundir/learnergy/
export LD_LIBRARY_PATH=${CUDA_HOME}:${LD_LIBRARY_PATH}

which torchrun
cd /app/rundir/SIT_FUSE/
torchrun --nnodes=1 --nproc_per_node=1 dbn_learnergy.py --yaml /app/rundir/SIT_FUSE/hpc/tacc_ls6/dbn/uavsar_dbn.yaml




