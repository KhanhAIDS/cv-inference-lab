import argparse
from pathlib import Path
from ultralytics import YOLO
from rich.console import Console

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO baseline for D-Fire")
    parser.add_argument("--data", type=str, required=True, help="Path to data YAML file")
    parser.add_argument("--model", type=str, required=True, help="YOLO model version (e.g. yolov8n.pt)")
    parser.add_argument("--run-name", type=str, required=True, help="Name for the training run")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    return parser.parse_args()

def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    
    if not data_path.exists():
        console.print(f"[red]Data file {data_path} not found![/red]")
        return
        
    try:
        model = YOLO(args.model)
        console.print(f"[green]Successfully loaded {args.model}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load model {args.model}. Error: {e}[/red]")
        return
        
    project_dir = Path("artifacts/runs").resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[green]Starting training run {args.run_name}...[/green]")
    
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        patience=20,
        batch=-1,  # auto batch if supported, fallback handles internally or user can override
        imgsz=640,
        seed=20260707,
        pretrained=True,
        project=str(project_dir),
        name=args.run_name,
        exist_ok=True
    )
    
    console.print(f"[green]Training completed! Results saved to {project_dir / args.run_name}[/green]")

if __name__ == "__main__":
    main()
