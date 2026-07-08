import argparse
import json
import platform
import sys
from pathlib import Path

from rich.console import Console
from ultralytics import YOLO

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--run-name", default="dfire_yolo26n_baseline")
    parser.add_argument("--project", default="artifacts/smoke_fire_detection/runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=float, default=0.95)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--device")
    parser.add_argument("--metadata-out")
    parser.add_argument("--resume", choices=["auto", "always", "never"], default="auto")
    return parser.parse_args()

def should_resume(mode: str, last_path: Path):
    if mode == "never":
        if last_path.exists():
            raise RuntimeError(f"Checkpoint exists, refusing fresh train in existing run: {last_path}")
        return False
    if last_path.exists():
        return True
    if mode == "always":
        raise FileNotFoundError(last_path)
    return False

def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    run_dir = project_dir / args.run_name
    last_path = run_dir / "weights" / "last.pt"
    resume_used = should_resume(args.resume, last_path)
    if resume_used:
        model = YOLO(str(last_path))
        train_kwargs = {"resume": True}
        if args.device:
            train_kwargs["device"] = args.device
    else:
        model = YOLO(args.model)
        train_kwargs = {
            "data": str(data_path),
            "epochs": args.epochs,
            "patience": args.patience,
            "save_period": args.save_period,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "seed": args.seed,
            "pretrained": True,
            "project": str(project_dir),
            "name": args.run_name,
            "exist_ok": True,
        }
        if args.device:
            train_kwargs["device"] = args.device
    model.train(**train_kwargs)
    if args.metadata_out:
        metadata_path = Path(args.metadata_out).resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "command": sys.argv,
            "data": str(data_path),
            "model": args.model,
            "run_dir": str(run_dir),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "save_period": args.save_period,
            "seed": args.seed,
            "device": args.device or "auto",
            "resume": args.resume,
            "resume_used": resume_used,
            "last_checkpoint": str(last_path),
            "python": sys.version,
            "platform": platform.platform(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    console.print(f"Training saved: {run_dir}")

if __name__ == "__main__":
    main()
