import argparse
import json
import time
from itertools import cycle, islice
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from ultralytics import YOLO

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model (accuracy and latency)")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default="artifacts/smoke_fire_detection/eval_report.json")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--measured", type=int, default=500)
    return parser.parse_args()

def split_images(data_yaml: Path, split: str):
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(data.get("path", data_yaml.parent)).resolve()
    split_file = data.get(split)
    if not split_file:
        return []
    split_file = Path(split_file)
    if not split_file.is_absolute():
        split_file = base / split_file
    if not split_file.exists():
        return []
    return [Path(line.strip()) for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    data_path = Path(args.data).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    model = YOLO(weights_path)
    
    # 1. Accuracy metrics
    console.print("Starting accuracy evaluation...")
    metrics = model.val(data=str(data_path), split=args.split, conf=args.conf, iou=args.iou, save_json=True)
    
    results = {
        "split": args.split,
        "weights": str(weights_path),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "classes": {},
    }
    for index, class_id in enumerate(metrics.box.ap_class_index):
        name = model.names[int(class_id)]
        results["classes"][name] = {
            "mAP50": float(metrics.box.map50s[index]),
            "mAP50_95": float(metrics.box.maps[index]),
        }

    # 2. Latency/Throughput Profiling
    console.print("Starting latency profiling...")
    images = split_images(data_path, args.split)
    if images:
        for image_path in islice(cycle(images), args.warmup):
            model(str(image_path), verbose=False)
    
        latencies = []
        for image_path in islice(cycle(images), args.measured):
            start = time.perf_counter()
            model(str(image_path), verbose=False)
            latencies.append((time.perf_counter() - start) * 1000)
    
        results.update({
            "samples": args.measured,
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "fps": float(1000 / np.mean(latencies)),
        })
    else:
        console.print("[yellow]Warning: No images found for latency profiling.[/yellow]")

    # Output
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(f"Eval report saved: {out_path}")

if __name__ == "__main__":
    main()
