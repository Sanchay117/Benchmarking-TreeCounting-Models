#!/bin/bash
set -e

PYTHON_BIN="/media/NAS/ashank/conda_envs/prithvi_env/bin/python"

cd /home/ashank/TreeCounting_Benchmark/anysat_tinytrees

for VARIANT in anysat anysat_full; do
    echo "============================================="
    echo "Evaluating $VARIANT"
    echo "============================================="

    # Dynamically find the latest checkpoint directory (ignoring the 001455 test run if any)
    PS_DIR=$(ls -d checkpoints/${VARIANT}_ps_* | tail -n 1)
    GF_DIR=$(ls -d checkpoints/${VARIANT}_gf_* | tail -n 1)
    SPOT_DIR=$(ls -d checkpoints/${VARIANT}_spot_* | tail -n 1)

    PS_CHECKPOINT="${PS_DIR}/checkpoint_best.pth"
    GF_CHECKPOINT="${GF_DIR}/checkpoint_best.pth"
    SPOT_CHECKPOINT="${SPOT_DIR}/checkpoint_best.pth"

    echo "Evaluating $VARIANT on PlanetScope using $PS_CHECKPOINT..."
    $PYTHON_BIN -u test_anysat.py --sensor ps --model_variant $VARIANT --checkpoint $PS_CHECKPOINT > "logs_eval_${VARIANT}_ps.txt"

    echo "Evaluating $VARIANT on Gaofen-2 using $GF_CHECKPOINT..."
    $PYTHON_BIN -u test_anysat.py --sensor gf --model_variant $VARIANT --checkpoint $GF_CHECKPOINT > "logs_eval_${VARIANT}_gf.txt"

    echo "Evaluating $VARIANT on SPOT-6 using $SPOT_CHECKPOINT..."
    $PYTHON_BIN -u test_anysat.py --sensor spot --model_variant $VARIANT --checkpoint $SPOT_CHECKPOINT > "logs_eval_${VARIANT}_spot.txt"
done

echo "All evaluations completed!"
