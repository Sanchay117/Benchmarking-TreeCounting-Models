from hub_model import UNetR50
import torch

for sensor in ["ps", "gf", "spot"]:
    model = UNetR50.from_pretrained(f"./pretrained_models/{sensor}")
    print(f"Loaded {sensor}, sum of weights:", sum(p.sum().item() for p in model.parameters()))
