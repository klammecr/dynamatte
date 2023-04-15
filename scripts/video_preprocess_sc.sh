# Stealing parts of the script from the PHANG repo (mine)

#!/bin/bash
#SBATCH --job-name=dynamatte_preprocess
#SBATCH --output=logs.dynamatte_pp.out
#SBATCH --error=logs.dynamatte_pp.err
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --mem=60GB
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --partition=dev

set -x
srun python -u video_preprocessing/process_video.py