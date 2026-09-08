#!/bin/bash
set -e

PYTHON_BIN="/media/NAS/ashank/conda_envs/prithvi_env/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODELS=("galileo_nano" "galileo_tiny" "galileo_base")
RATIOS=(1.0 0.8)
SENSORS=("ps" "gf" "spot")

echo "==========================================================="
echo "   Galileo Foundation Model Evaluation Pipeline           "
echo "==========================================================="

for MODEL in "${MODELS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        WEAK_PCT=$(python -c "print(int(round((1.0 - $RATIO) * 100)))")
        for SENSOR in "${SENSORS[@]}"; do
            PATTERN="checkpoints/${MODEL}_${SENSOR}_weak${WEAK_PCT}_*"
            MATCHING_DIRS=($(ls -d $PATTERN 2>/dev/null | sort -V))

            if [ ${#MATCHING_DIRS[@]} -eq 0 ]; then
                echo "No checkpoint found for $MODEL on $SENSOR (weak ${WEAK_PCT}%). Skipping."
                continue
            fi

            LATEST_DIR="${MATCHING_DIRS[-1]}"
            CHECKPOINT="${LATEST_DIR}/checkpoint_best.pth"

            if [ ! -f "$CHECKPOINT" ]; then
                CHECKPOINT="${LATEST_DIR}/latest.pth"
            fi

            if [ ! -f "$CHECKPOINT" ]; then
                echo "Warning: Checkpoint file not found in $LATEST_DIR. Skipping."
                continue
            fi

            LOG_FILE="logs_eval_${MODEL}_${SENSOR}_weak${WEAK_PCT}.txt"
            echo "Evaluating $MODEL on $SENSOR (weak ${WEAK_PCT}%) using $CHECKPOINT..."
            $PYTHON_BIN -u test_galileo.py \
                --sensor "$SENSOR" \
                --model_variant "$MODEL" \
                --checkpoint "$CHECKPOINT" \
                --batch_size 16 \
                --patch_size 4 > "$LOG_FILE" 2>&1

            echo "Output saved to $LOG_FILE"
            cat "$LOG_FILE" | grep -A 6 "=== Test Results ===" || true
            echo ""
        done
    done
done

echo "==========================================================="
echo "All available Galileo evaluations completed!"
echo "Updating results markdown..."
$PYTHON_BIN update_results.py || true
echo "==========================================================="
