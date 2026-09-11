#!/bin/bash
set -e

PYTHON_BIN="/media/NAS/ashank/conda_envs/prithvi_env/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODELS=("anysat" "anysat_full")
RATIOS=(1.0 0.8)
SENSORS=("ps" "gf" "spot")

# Optional CLI arguments to override: ./train_all.sh [model] [ratio] [sensor]
TARGET_MODEL=${1:-"all"}
TARGET_RATIO=${2:-"all"}
TARGET_SENSOR=${3:-"all"}

echo "==========================================================="
echo "   AnySat Foundation Model Fine-Tuning Pipeline            "
echo "==========================================================="

for MODEL in "${MODELS[@]}"; do
    if [ "$TARGET_MODEL" != "all" ] && [ "$TARGET_MODEL" != "$MODEL" ]; then
        continue
    fi

    for RATIO in "${RATIOS[@]}"; do
        if [ "$TARGET_RATIO" != "all" ] && [ "$TARGET_RATIO" != "$RATIO" ]; then
            continue
        fi

        WEAK_PCT=$(python -c "print(int(round((1.0 - $RATIO) * 100)))")
        if [ "$RATIO" == "1.0" ]; then
            SUP_LABEL="100% Strong"
        else
            SUP_LABEL="80% Strong + 20% Weak"
        fi

        for SENSOR in "${SENSORS[@]}"; do
            if [ "$TARGET_SENSOR" != "all" ] && [ "$TARGET_SENSOR" != "$SENSOR" ]; then
                continue
            fi

            echo "-----------------------------------------------------------"
            echo "Training $MODEL on $SENSOR ($SUP_LABEL, weak=${WEAK_PCT}%)"
            echo "-----------------------------------------------------------"

            $PYTHON_BIN -u train_anysat.py \
                --sensor "$SENSOR" \
                --model_variant "$MODEL" \
                --strong_ratio "$RATIO" \
                --epochs 50 \
                --batch_size 16 \
                --lr 1e-5
        done
    done
done

echo "==========================================================="
echo "All requested AnySat training runs completed successfully!"
echo "==========================================================="
