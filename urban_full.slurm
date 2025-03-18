#! /bin/sh

#SBATCH --job-name=full_urban
#SBATCH --output=/home/yandex/MLWG2024/omertalmi/audio/Audio_Project/full_urban.out # redirect stdout
#SBATCH --error=/home/yandex/MLWG2024/omertalmi/audio/Audio_Project/full_urban.err # redirect stderr
#SBATCH --partition=studentkillable  # (see resources section)
#SBATCH --account=gpu-students
#SBATCH --time=500 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=2 # GPUs in total

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Start"
# unset PYTORCH_CUDA_ALLOC_CONF
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
python3 main.py --data_path dataset/ --is_AST True --dataset_name 'urbansound8k' --method 'full-FT'
echo "Finish"

