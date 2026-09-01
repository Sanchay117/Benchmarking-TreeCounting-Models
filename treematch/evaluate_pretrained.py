import torch
from torch.utils.data import DataLoader
from hub_model import UNetR50
from data.ps import PlanetScopeStrong
from data.gf import GaofenStrong
from data.spot import SPOTStrong
from train import get_preds, evaluate_at_fixed_scale

def evaluate_model(sensor, dataset_class, root_dir, gsd_m):
    print(f"\n--- Evaluating {sensor.upper()} ---")
    
    # Load model from local downloaded directory
    model = UNetR50.from_pretrained(f"./pretrained_models/{sensor}")
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load test split
    test_ds = dataset_class(imsize=64, split="test", root=root_dir)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # Create dummy trainer to use get_preds
    class DummyTrainer:
        def __init__(self, model):
            self.backbone = model
        def predict(self, inputs):
            inputs = inputs.to(device)
            valid = inputs[:, [-1,]].to(device)
            outputs = torch.nn.functional.relu(self.backbone(inputs)) * valid
            return outputs
        
    trainer = DummyTrainer(model)

    # Get predictions
    test_pred, test_target, test_valid = get_preds(test_loader, trainer, device)
    
    # Calculate metrics
    metrics = evaluate_at_fixed_scale(
        test_pred, test_target, gsd_m=gsd_m, masks=test_valid, eval_patch_size=64
    )
    
    print(f"Results for {sensor.upper()}:")
    print(f"  RMSE: {metrics.get('rmse', 'N/A'):.2f}")
    print(f"  R^2 : {metrics.get('r2', 'N/A'):.2f}")
    print(f"  nMAE: {metrics.get('nmae', 'N/A'):.2f}")

if __name__ == "__main__":
    dataset_base = "/home/ashank/TreeCounting_Benchmark/tinytrees_dataset/tinytrees"
    
    # Evaluate PlanetScope (Rwanda)
    evaluate_model("ps", PlanetScopeStrong, f"{dataset_base}/ps", gsd_m=3.0)
    
    # Evaluate Gaofen-2 (China)
    evaluate_model("gf", GaofenStrong, f"{dataset_base}/gf", gsd_m=0.8)
    
    # Evaluate SPOT-6 (France)
    evaluate_model("spot", SPOTStrong, f"{dataset_base}/spot", gsd_m=1.5)
