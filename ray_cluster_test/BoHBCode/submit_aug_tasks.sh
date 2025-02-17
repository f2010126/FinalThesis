#!/bin/bash -l

#SBATCH --partition=mlhiwidlc_gpu-rtx2080-advanced
#SBATCH --job-name=GenAugTasks

### This script works for any number of nodes
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32

#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/dlclarge1/dsengupt-zap_hpo_og/logs/GenAugTasks.out
#SBATCH --error=/work/dlclarge1/dsengupt-zap_hpo_og/logs/GenAugTasks.error

#SBATCH --time=00:40:00

# set up env
source ~/ray_env/bin/activate
cd $(ws_find zap_hpo_og)/TinyBert/HPO/ray_cluster_test/BoHBCode

filtered_tasks=$(grep 'Bundestag-v2_1X' datasets.txt)

# Loop through each filtered task name
while IFS= read -r task_name; do
    # Run the Python script with the filtered task name
    python bohb_ray_slurm_launch.py --exp-name BohbAugmented \
                                    --num-nodes 2 \
                                    --num-gpus 8 \
                                    --partition mlhiwidlc_gpu-rtx2080-advanced \
                                    --runtime 14:20:00 \
                                    --task-name "$task_name" \
                                    --n_workers 4 \
                                    --eta 2 \
                                    --max_budget 5 \
                                    --n_iter 25 \
                                    --w_gpu 4 \
                                    --aug \
                                    --prev_run Bundestag-v2_Bohb_P2_5_2
done <<< "$filtered_tasks"

echo "All tasks submitted"