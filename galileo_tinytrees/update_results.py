import os
import glob
import re

MD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../treematch/treematch_evaluation.md"))
LOG_DIR = os.path.dirname(__file__)

SENSOR_NAME_MAP = {
    "gf": "Gaofen-2",
    "ps": "PlanetScope",
    "spot": "SPOT-6"
}

MODEL_NAME_MAP = {
    "galileo_nano": "Galileo-Nano",
    "galileo_tiny": "Galileo-Tiny",
    "galileo_base": "Galileo-Base"
}

def parse_log_file(filepath):
    # Pattern: logs_eval_{model}_{sensor}_weak{weak_pct}.txt
    filename = os.path.basename(filepath)
    m = re.match(r"logs_eval_(galileo_\w+)_(ps|gf|spot)_weak(\d+)\.txt", filename)
    if not m:
        return None
    model, sensor, weak_pct = m.groups()
    weak_pct = int(weak_pct)
    sup_label = "100% Strong" if weak_pct == 0 else f"{100 - weak_pct}% Strong + {weak_pct}% Weak"

    with open(filepath, "r") as f:
        content = f.read()

    rmse, r2, nmae, mae = None, None, None, None
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("RMSE:"):
            rmse = float(line.split(":")[1].strip())
        elif line.startswith("R2:"):
            r2 = float(line.split(":")[1].strip())
        elif line.startswith("NMAE:"):
            # Can be e.g. "55.20% (raw: 0.5520)" or "0.5520"
            val_part = line.split(":")[1].strip()
            if "%" in val_part:
                nmae = float(val_part.split("%")[0].strip())
            else:
                nmae = float(val_part) * 100.0
        elif line.startswith("MAE:"):
            mae = float(line.split(":")[1].strip())

    if rmse is not None and r2 is not None and nmae is not None and mae is not None:
        model_display = MODEL_NAME_MAP.get(model, model)
        sensor_display = SENSOR_NAME_MAP.get(sensor, sensor)
        return {
            "key": (model, weak_pct, sensor),
            "model_sensor": f"**{model_display} / {sensor_display}**",
            "supervision": sup_label,
            "rmse": f"{rmse:.2f}",
            "r2": f"{r2:.2f}",
            "nmae": f"{nmae:.1f}",
            "mae": f"{mae:.2f}"
        }
    return None

def main():
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "logs_eval_galileo_*.txt")))
    results = []
    for f in log_files:
        parsed = parse_log_file(f)
        if parsed:
            results.append(parsed)

    if not results:
        print("No completed evaluation logs found yet.")
        return

    # Sort results nicely: by model order, then weak_pct, then sensor
    model_order = {"galileo_nano": 0, "galileo_tiny": 1, "galileo_base": 2}
    sensor_order = {"gf": 0, "ps": 1, "spot": 2}
    results.sort(key=lambda x: (
        model_order.get(x["key"][0], 99),
        sensor_order.get(x["key"][2], 99),
        x["key"][1]
    ))

    table_lines = [
        "### Galileo Foundation Model Fine-Tuning\n",
        "Below is the evaluation of the `Galileo` models fine-tuned on the TINYTREES benchmark using the TreeMatch framework (`treematch` uOT Trainer + `FCNDecoder`). The models were fine-tuned across both 100% Strong and 80% Strong + 20% Weak supervision regimes. (Note: The nMAE outputs from the script have been converted to percentages to match the tables above).\n",
        "| Model / Sensor | Supervision Used | RMSE (↓) | R² (↑) | nMAE (↓) | MAE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for r in results:
        table_lines.append(f"| {r['model_sensor']} | {r['supervision']} | {r['rmse']} | {r['r2']} | {r['nmae']} | {r['mae']} |")

    new_section = "\n".join(table_lines) + "\n"

    with open(MD_PATH, "r") as f:
        md_content = f.read()

    section_header = "### Galileo Foundation Model Fine-Tuning"
    if section_header in md_content:
        # Replace existing section
        parts = md_content.split(section_header)
        md_content = parts[0].rstrip() + "\n\n" + new_section
    else:
        # Append to the end
        md_content = md_content.rstrip() + "\n\n" + new_section

    with open(MD_PATH, "w") as f:
        f.write(md_content)

    print(f"Updated {MD_PATH} with {len(results)} Galileo benchmark results successfully!")

if __name__ == "__main__":
    main()
