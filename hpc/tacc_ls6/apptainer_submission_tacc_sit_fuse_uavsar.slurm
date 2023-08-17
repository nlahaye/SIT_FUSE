#!/bin/bash -l

#
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=output_sit_fuse.out

module load tacc-apptainer
module load cuda/11.4
module load nccl/2.11.4
module load cudnn/8.2.4

/usr/bin/nvidia-smi
cd $WORK/SIT_FUSE/hpc/tacc_ls6/
apptainer run --bind /work/09562/nleet/ls6/:/app/rundir/ --nv ./sit_fuse.sif
cd $WORK/SIT_FUSE
export PYTHONPATH=$PYTHONPATH:$WORK/learnergy/:$WORK/SIT_FUSE/
torchrun --nnodes=1 --nproc_per_node=1 dbn_learergy.py --yaml $WORK/config/dbn/dbn_config.yaml