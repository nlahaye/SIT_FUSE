#!/bin/bash -l
#
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --output=test-srun_sing.out

#module purge
#module load conda
#module spider
eval "$(conda shell.bash hook)"
#module load cuda/11.6.1


module load cuda
module load conda
/usr/bin/nvidia-smi
cd ~/MultiModalObj-master/
singularity exec --nv -B /home/hack_team_12/:/home/hack_team_12/ /home/mz23hu84/pythorch_22.08.simg ./container_script.sh

#torchrun

