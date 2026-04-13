import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets.mcnn_kcl import KCLLondonMCNNDataset
from network.csrnet import CSRNet


def save_hyperparams(save_dir: str, args_dict: dict):
    with open(os.path.join(save_dir, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, sort_keys=True)


def save_history_csv(save_dir: str, history: list):
    csv_path = os.path.join(save_dir, "history.csv")
    fieldnames = ["epoch", "train_mse", "val_mae", "val_mse"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_history_json(save_dir: str, history: list, best_mae: float):
    payload = {
        "best_mae": (None if np.isinf(best_mae) else float(best_mae)),
        "history": history,
    }
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_plots(save_dir: str, history: list):
    if len(history) == 0:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib is unavailable or incompatible ({exc}). Skipping graph export.")
        return

    epochs = [row["epoch"] for row in history]
    eps = 1e-8
    train_mse = [max(float(row["train_mse"]), eps) for row in history]

    val_points = [row for row in history if row["val_mae"] is not None and row["val_mse"] is not None]
    val_epochs = [row["epoch"] for row in val_points]
    val_mae = [max(float(row["val_mae"]), eps) for row in val_points]
    val_mse = [max(float(row["val_mse"]), eps) for row in val_points]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_mse, label="Train MSE", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.yscale("log")
    plt.title("CSRNet Training MSE (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_mse_curve.png"), dpi=150)
    plt.close()

    if len(val_points) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(val_epochs, val_mae, label="Val MAE", linewidth=2)
        plt.plot(val_epochs, val_mse, label="Val MSE", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Error (log scale)")
        plt.yscale("log")
        plt.title("CSRNet Validation Curves (Log Scale)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "val_curves.png"), dpi=150)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CSRNet on KCL London dataset")
    parser.add_argument("--data-dir", type=str, default="datasets", help="Dataset root inside TreeFormer")
    parser.add_argument("--train-split", type=str, default="train_data", help="Training split folder")
    parser.add_argument("--val-split", type=str, default="valid_data", help="Validation split folder")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--crop-size", type=int, default=256, help="Training crop size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id, e.g. 0")
    parser.add_argument("--seed", type=int, default=64678, help="Random seed")
    parser.add_argument("--val-every", type=int, default=1, help="Validate every N epochs")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="Optional CSRNet checkpoint (.pth/.pt)",
    )
    parser.add_argument(
        "--no-vgg-pretrain",
        action="store_true",
        help="Disable ImageNet VGG16 initialization for CSRNet frontend",
    )
    parser.add_argument("--save-dir", type=str, default="ckpts/csrnet_kcl", help="Checkpoint output directory")
    return parser.parse_args()


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_legacy_h5(path: str, model: nn.Module):
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


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    if not path:
        return 0
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.endswith(".h5"):
        load_legacy_h5(path, model)
        return 0
    if path.endswith(".caffemodel"):
        raise ValueError(
            "Direct .caffemodel loading is not supported in this PyTorch CSRNet trainer. "
            "Convert it to .pth first or start without --pretrained."
        )

    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return int(ckpt.get("epoch", -1)) + 1

    if isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=True)
    else:
        raise ValueError("Unsupported checkpoint format")
    return 0


def resize_density_to_output(gt_density: torch.Tensor, pred_density: torch.Tensor) -> torch.Tensor:
    _, _, in_h, in_w = gt_density.shape
    _, _, out_h, out_w = pred_density.shape
    if (in_h, in_w) == (out_h, out_w):
        return gt_density

    gt_resized = F.interpolate(gt_density, size=(out_h, out_w), mode="bilinear", align_corners=False)
    scale = float(in_h * in_w) / float(out_h * out_w)
    return gt_resized * scale


def evaluate(model, loader, device):
    model.eval()
    abs_err = []
    sq_err = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            image = image.repeat(1, 3, 1, 1)
            gt_count = batch["count"].to(device, non_blocking=True)

            pred_density = model(image)
            pred_count = pred_density.flatten(1).sum(dim=1)

            err = gt_count - pred_count
            abs_err.append(torch.abs(err))
            sq_err.append(err * err)

    abs_err = torch.cat(abs_err)
    sq_err = torch.cat(sq_err)
    mae = abs_err.mean().item()
    mse = torch.sqrt(sq_err.mean()).item()
    return mae, mse


def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)

    train_dataset = KCLLondonMCNNDataset(
        root=args.data_dir,
        split=args.train_split,
        crop_size=args.crop_size,
        random_flip=True,
    )
    val_dataset = KCLLondonMCNNDataset(
        root=args.data_dir,
        split=args.val_split,
        crop_size=None,
        random_flip=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = CSRNet(load_frontend_pretrained=(not args.no_vgg_pretrain)).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = load_checkpoint(args.pretrained, model, optimizer=optimizer)

    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, run_stamp)
    os.makedirs(save_dir, exist_ok=True)
    save_hyperparams(save_dir, vars(args))

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Saving checkpoints to: {save_dir}")

    best_mae = float("inf")
    history = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            image = image.repeat(1, 3, 1, 1)
            gt_density = batch["density"].to(device, non_blocking=True)

            pred_density = model(image)
            gt_down = resize_density_to_output(gt_density, pred_density)
            loss = criterion(pred_density, gt_down)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * image.size(0)

        epoch_loss /= max(1, len(train_dataset))
        print(f"Epoch {epoch + 1}/{args.epochs} - train_mse: {epoch_loss:.6f}")

        epoch_record = {
            "epoch": epoch + 1,
            "train_mse": float(epoch_loss),
            "val_mae": None,
            "val_mse": None,
        }

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(save_dir, "latest.pth"))

        if (epoch + 1) % args.val_every != 0:
            history.append(epoch_record)
            save_history_csv(save_dir, history)
            save_history_json(save_dir, history, best_mae)
            save_plots(save_dir, history)
            continue

        mae, mse = evaluate(model, val_loader, device)
        epoch_record["val_mae"] = float(mae)
        epoch_record["val_mse"] = float(mse)
        history.append(epoch_record)

        save_history_csv(save_dir, history)
        save_history_json(save_dir, history, best_mae)
        save_plots(save_dir, history)

        print(f"Epoch {epoch + 1}/{args.epochs} - val_mae: {mae:.4f} - val_mse: {mse:.4f}")

        if mae < best_mae:
            best_mae = mae
            torch.save(ckpt, os.path.join(save_dir, "best_mae.pth"))
            save_history_json(save_dir, history, best_mae)
            print(f"Saved new best checkpoint with MAE {best_mae:.4f}")


if __name__ == "__main__":
    main()
