#!/bin/bash
set -e

PYTHON_BIN="/media/NAS/ashank/conda_envs/prithvi_env/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODELS=("anysat" "anysat_full")
RATIOS=(1.0 0.8)
SENSORS=("ps" "gf" "spot")

echo "==========================================================="
echo "   AnySat Foundation Model Evaluation Pipeline             "
echo "==========================================================="

for MODEL in "${MODELS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        WEAK_PCT=$(python -c "print(int(round((1.0 - $RATIO) * 100)))")
        for SENSOR in "${SENSORS[@]}"; do
            # Check for pattern with weak_pct first, then fallback to legacy naming if weak_pct is 0
            if [ "$WEAK_PCT" -eq 0 ]; then
                MATCHING_DIRS=($(ls -d checkpoints/${MODEL}_${SENSOR}_weak0_* 2>/dev/null | sort -V))
                if [ ${#MATCHING_DIRS[@]} -eq 0 ]; then
                    MATCHING_DIRS=($(ls -d checkpoints/${MODEL}_${SENSOR}_[0-9]* 2>/dev/null | sort -V))
                fi
            else
                MATCHING_DIRS=($(ls -d checkpoints/${MODEL}_${SENSOR}_weak${WEAK_PCT}_* 2>/dev/null | sort -V))
            fi

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
            $PYTHON_BIN -u test_anysat.py \
                --sensor "$SENSOR" \
                --model_variant "$MODEL" \
                --checkpoint "$CHECKPOINT" \
                --batch_size 16 > "$LOG_FILE" 2>&1

            echo "Output saved to $LOG_FILE"
            cat "$LOG_FILE" | grep -A 6 "=== Test Results ===" || true
            echo ""
        done
    done
done

echo "==========================================================="
echo "All available AnySat evaluations completed!"
echo "Updating results markdown..."
$PYTHON_BIN update_results.py || true
echo "==========================================================="
