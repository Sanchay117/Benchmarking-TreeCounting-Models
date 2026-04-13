import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from scipy.io import savemat

from datasets.mcnn_kcl import KCLLondonMCNNDataset
from network.csrnet import CSRNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned CSRNet checkpoint on KCL London")
    parser.add_argument("--data-dir", type=str, default="datasets", help="Dataset root inside TreeFormer")
    parser.add_argument("--split", type=str, default="test_data", help="Dataset split to evaluate")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a saved CSRNet checkpoint")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id, e.g. 0")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--output-dir", type=str, default="eval", help="Root directory for evaluation runs")
    parser.add_argument("--pred-dir", type=str, default="predictions_csrnet", help="Directory for saved .mat predictions")
    return parser.parse_args()


def prepare_eval_run_dir(eval_root: str, prefix: str) -> str:
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(eval_root, f"{prefix}_{run_stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_hyperparams(run_dir: str, args):
    hparams_path = os.path.join(run_dir, "hparams.json")
    with open(hparams_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    return hparams_path


def resolve_device(device_arg: str) -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = device_arg
    if not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    arch_list = set(torch.cuda.get_arch_list())
    major, minor = torch.cuda.get_device_capability(0)
    sm = f"sm_{major}{minor}"
    if sm not in arch_list:
        print(
            f"CUDA arch {sm} is not supported by this PyTorch build ({sorted(arch_list)}). "
            "Falling back to CPU."
        )
        return torch.device("cpu")

    return torch.device("cuda")


def load_legacy_h5(path: str, model: torch.nn.Module):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to load .h5 checkpoints") from exc

    loaded = 0
    state = model.state_dict()
    with h5py.File(path, "r") as h5f:
        for key in state:
            candidates = [key, f"DME.{key}"]
            selected = None
            for candidate in candidates:
                if candidate in h5f:
                    selected = candidate
                    break
            if selected is None:
                continue

            tensor = torch.from_numpy(np.asarray(h5f[selected]))
            if tensor.shape != state[key].shape:
                continue
            state[key].copy_(tensor)
            loaded += 1

    if loaded == 0:
        raise ValueError(f"No compatible tensors were loaded from legacy checkpoint: {path}")
    print(f"Loaded {loaded}/{len(state)} tensors from legacy checkpoint: {path}")


def load_checkpoint(model: torch.nn.Module, model_path: str, device: torch.device):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if model_path.endswith(".h5"):
        load_legacy_h5(model_path, model)
        return -1
    if model_path.endswith(".caffemodel"):
        raise ValueError(
            "Direct .caffemodel loading is not supported in this PyTorch CSRNet evaluator. "
            "Use a .pth checkpoint from train_csrnet_kcl.py."
        )

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return int(checkpoint.get("epoch", -1))

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return int(checkpoint.get("epoch", -1))

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=True)
        return -1

    raise ValueError("Unsupported checkpoint format")


def compute_r2_score(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.size == 0:
        return 0.0

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def evaluate(args):
    device = resolve_device(args.device)
    run_dir = prepare_eval_run_dir(args.output_dir, "csrnet_test")
    hparams_path = save_hyperparams(run_dir, args)
    os.makedirs(args.pred_dir, exist_ok=True)

    dataset = KCLLondonMCNNDataset(root=args.data_dir, split=args.split, crop_size=None, random_flip=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = CSRNet(load_frontend_pretrained=False).to(device)
    load_checkpoint(model, args.model_path, device)
    model.eval()

    abs_errors = []
    pred_counts = []
    gt_counts = []
    count_errors = []
    results = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            image = image.repeat(1, 3, 1, 1)
            gt_density = batch["density"].to(device, non_blocking=True)
            gt_count = batch["count"].to(device, non_blocking=True)
            name = batch["name"]

            pred_density = model(image)
            pred_count = pred_density.flatten(1).sum(dim=1)

            err = gt_count - pred_count
            abs_errors.append(torch.abs(err))
            count_errors.extend(err.detach().cpu().tolist())
            pred_counts.extend(pred_count.detach().cpu().tolist())
            gt_counts.extend(gt_count.detach().cpu().tolist())

            name_str = name[0] if isinstance(name, (list, tuple)) else name
            pred_val = float(pred_count.item())
            gt_val = float(gt_count.item())
            err_val = gt_val - pred_val
            print(f"Img name: {name_str} Error: {err_val:.4f} GT count: {gt_val:.4f} Model out: {pred_val:.4f}")
            results.append([name_str, gt_val, pred_val, err_val])

            density_cpu = pred_density[0].detach().cpu().numpy()
            image_cpu = image[0].detach().cpu().numpy()
            gt_density_cpu = gt_density[0].detach().cpu().numpy()
            savemat(
                os.path.join(args.pred_dir, f"{name_str}.mat"),
                {
                    "estimation": np.squeeze(density_cpu),
                    "image": np.squeeze(image_cpu),
                    "gt": np.squeeze(gt_density_cpu),
                },
            )

    abs_errors = torch.cat(abs_errors)
    count_errors = np.asarray(count_errors, dtype=np.float32)
    mse = float(np.sqrt(np.mean(np.square(count_errors))))
    mae = float(abs_errors.mean().item())
    r2 = compute_r2_score(gt_counts, pred_counts)

    print(f"{args.model_path}: mae {mae}, mse {mse}, R2 {r2}\n")

    final_stats = {
        "model_path": args.model_path,
        "split": args.split,
        "num_images": int(len(results)),
        "mae": mae,
        "mse": mse,
        "r2": r2,
    }
    final_stats_path = os.path.join(run_dir, "final_stats.json")
    with open(final_stats_path, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, indent=2)

    text_lines = [str(row).replace("[", "").replace("]", "").replace(",", " ") + "\n" for row in results]
    summary_path = os.path.join(run_dir, "test_csrnet_kcl.txt")
    legacy_path = os.path.join(run_dir, "test.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(text_lines)

    with open(legacy_path, "w", encoding="utf-8") as f:
        f.writelines(text_lines)

    print(f"Saved hyperparameters: {hparams_path}")
    print(f"Saved eval run directory: {run_dir}")
    print(f"Saved final stats: {final_stats_path}")
    print(f"Saved .mat predictions to: {args.pred_dir}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved summary: {legacy_path}")


if __name__ == "__main__":
    evaluate(parse_args())
