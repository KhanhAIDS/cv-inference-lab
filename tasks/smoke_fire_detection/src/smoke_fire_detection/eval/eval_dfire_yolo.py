import argparse
import json
from pathlib import Path
from ultralytics import YOLO
from rich.console import Console

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO on D-Fire")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to data YAML file")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate on (val or test)")
    parser.add_argument("--out", type=str, default="artifacts/dfire/eval_results.json", help="Output path")
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    weights_path = Path(args.weights).resolve()
    out_path = Path(args.out).resolve()
    
    if not weights_path.exists():
        console.print(f"[red]Weights not found: {weights_path}[/red]")
        return
        
    model = YOLO(weights_path)
    console.print(f"[green]Evaluating on split: {args.split}[/green]")
    
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        conf=0.25,
        iou=0.6,
        save_json=True
    )
    
    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "classes": {}
    }
    
    for i, c in enumerate(metrics.box.ap_class_index):
        results["classes"][model.names[c]] = {
            "mAP50": float(metrics.box.map50s[i]),
            "mAP50-95": float(metrics.box.maps[i])
        }
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    console.print(f"[green]Results saved to {out_path}[/green]")
    
    # Optional: sweeping threshold logic and frame-level proxies could be added here
    # by running inference on the test split images directly.

if __name__ == "__main__":
    main()
