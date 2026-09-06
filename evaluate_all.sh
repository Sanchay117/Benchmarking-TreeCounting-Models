#!/bin/bash
cd /home/ashank/TreeCounting_Benchmark
OUTPUT_FILE="eval_results.md"
echo "| Model / Sensor | Supervision Used | RMSE (↓) | R² (↑) | nMAE (↓) | MAE |" > $OUTPUT_FILE
echo "| :--- | :---: | :---: | :---: | :---: | :---: |" >> $OUTPUT_FILE

for ckpt in prithvi_tinytrees/checkpoints/*/latest.pth; do
    folder=$(basename $(dirname $ckpt))
    
    # parse folder name: prithvi_vit_300_gf_weak0_2026-09-06_15-55-41
    IFS='_' read -ra PARTS <<< "$folder"
    model_name="${PARTS[0]}_${PARTS[1]}_${PARTS[2]}"
    sensor="${PARTS[3]}"
    weak="${PARTS[4]}"
    
    model_pretty="Prithvi-${PARTS[2]}M"
    
    if [ "$sensor" = "ps" ]; then sensor_pretty="PlanetScope"; fi
    if [ "$sensor" = "gf" ]; then sensor_pretty="Gaofen-2"; fi
    if [ "$sensor" = "spot" ]; then sensor_pretty="SPOT-6"; fi
    
    if [ "$weak" = "weak0" ]; then
        supervision="100% Strong"
    else
        supervision="80% Strong + 20% Weak"
    fi
    
    echo "Evaluating $folder..."
    
    OUT=$(/media/NAS/ashank/conda_envs/prithvi_env/bin/python prithvi_tinytrees/test_prithvi.py --model $model_name --sensor $sensor --checkpoint $ckpt 2>&1)
    
    r2=$(echo "$OUT" | grep "R2:" | awk '{print $2}')
    mae=$(echo "$OUT" | grep "MAE:" | awk '{print $2}')
    nmae=$(echo "$OUT" | grep "NMAE:" | awk '{print $2}')
    rmse=$(echo "$OUT" | grep "RMSE:" | awk '{print $2}')
    
    if [ -n "$r2" ]; then
        # Format nMAE to percentage (multiply by 100)
        nmae_pct=$(awk -v n="$nmae" 'BEGIN { printf "%.1f", n * 100 }')
        
        # Format to 2 decimals
        r2=$(awk -v n="$r2" 'BEGIN { printf "%.2f", n }')
        mae=$(awk -v n="$mae" 'BEGIN { printf "%.2f", n }')
        rmse=$(awk -v n="$rmse" 'BEGIN { printf "%.2f", n }')
        
        echo "| **$model_pretty / $sensor_pretty** | $supervision | $rmse | $r2 | $nmae_pct | $mae |" >> $OUTPUT_FILE
    else
        echo "Failed to parse $folder" >> $OUTPUT_FILE
        echo "$OUT" >> $OUTPUT_FILE
    fi
done
echo "Finished all evaluations!"
