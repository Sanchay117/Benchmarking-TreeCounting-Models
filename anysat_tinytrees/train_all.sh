#!/bin/bash
set -e

PYTHON_BIN="/media/NAS/ashank/conda_envs/prithvi_env/bin/python"

cd /home/ashank/TreeCounting_Benchmark/anysat_tinytrees

for VARIANT in anysat anysat_full; do
    echo "============================================="
    echo "Training $VARIANT"
    echo "============================================="

    # Train on PlanetScope
    echo "Training $VARIANT on PlanetScope..."
    $PYTHON_BIN -u train_anysat.py --sensor ps --model_variant $VARIANT --epochs 50

    # Train on Gaofen-2
    echo "Training $VARIANT on Gaofen-2..."
    $PYTHON_BIN -u train_anysat.py --sensor gf --model_variant $VARIANT --epochs 50

    # Train on SPOT-6
    echo "Training $VARIANT on SPOT-6..."
    $PYTHON_BIN -u train_anysat.py --sensor spot --model_variant $VARIANT --epochs 50
done

echo "All training completed!"
