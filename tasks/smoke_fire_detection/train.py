import argparse
from pathlib import Path

from rich.console import Console
from ultralytics import YOLO

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--run-name", default="yolo26n_baseline")
    parser.add_argument("--project", default="artifacts/smoke_fire_detection/runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        pretrained=True,
        project=str(project_dir),
        name=args.run_name,
        exist_ok=True,
    )
    console.print(f"Training saved: {project_dir / args.run_name}")

if __name__ == "__main__":
    main()
