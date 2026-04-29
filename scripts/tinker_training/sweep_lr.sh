#!/bin/bash
#
# Submit one SLURM job per learning rate.
# Usage: bash scripts/tinker_training/sweep_lr.sh

set -euo pipefail

EXPERIMENT="lr-sweep-sa"
BIAS_TYPES="suggested_answer"
DATASETS="mmlu,truthfulqa"
N_SAMPLES=100
MODEL="meta-llama/Llama-3.1-8B-Instruct"

LRS=(5e-5 1e-4 3e-4 5e-4)

mkdir -p logs/${EXPERIMENT}

for LR in "${LRS[@]}"; do
    # Build a run-name tag: 5e-5 -> lr5em05
    LR_TAG=$(echo "$LR" | sed 's/+//g; s/-/m/g; s/\.//g')
    RUN_NAME="rlct-${LR_TAG}"

    echo "Submitting LR=${LR}  run_name=${RUN_NAME}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --account=eecs498f25s017_class
#SBATCH --job-name=lr-${LR_TAG}
#SBATCH --mail-user=prakharg@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --output=logs/${EXPERIMENT}/${RUN_NAME}-slurm-%j.log

cd /home/prakharg/cot-transparency
source /home/prakharg/miniconda3/etc/profile.d/conda.sh
conda activate cot

python scripts/tinker_training/train_rl.py \
    --bias-types ${BIAS_TYPES} \
    --datasets ${DATASETS} \
    --n-samples ${N_SAMPLES} \
    --experiment-name ${EXPERIMENT} \
    --run-name ${RUN_NAME} \
    --model ${MODEL} \
    --lr ${LR} \
    --lr-schedule constant \
    --lora-rank 8 \
    --kl-coef 0.05 \
    --loss-fn ppo \
    --n-ref-samples 128 \
    --n-train-samples 128 \
    --n-grad-samples 128 \
    --temperature 1.0 \
    --n-epochs 1 \
    --situations-per-group 1 \
    --gradient-accumulation-steps 1 \
    --refresh-every 1 \
    --checkpoint-every 50 \
    -y
EOF
done

echo ""
echo "Submitted ${#LRS[@]} jobs. Monitor with: squeue -u \$USER"
