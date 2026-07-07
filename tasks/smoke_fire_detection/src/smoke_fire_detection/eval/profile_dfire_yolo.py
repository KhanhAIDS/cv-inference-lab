import argparse
import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from rich.console import Console

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Profile YOLO latency on D-Fire")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to data YAML file")
    parser.add_argument("--split", type=str, default="test", help="Split to profile on (val or test)")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup frames")
    parser.add_argument("--measured", type=int, default=500, help="Number of frames to measure")
    return parser.parse_args()

def main():
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    
    if not weights_path.exists():
        console.print(f"[red]Weights not found: {weights_path}[/red]")
        return
        
    model = YOLO(weights_path)
    
    # Actually, we need image paths from the split. 
    # For profiling, we can just use a dummy image of shape (640, 640, 3) 
    # or load real images if we want to include preprocess time.
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    console.print(f"[green]Warming up for {args.warmup} iterations...[/green]")
    for _ in range(args.warmup):
        _ = model(dummy_img, verbose=False)
        
    console.print(f"[green]Measuring latency for {args.measured} iterations...[/green]")
    latencies = []
    
    for _ in range(args.measured):
        start = time.perf_counter()
        _ = model(dummy_img, verbose=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1000) # in ms
        
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    mean_lat = np.mean(latencies)
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0
    
    console.print("[bold cyan]Latency Profile (ms):[/bold cyan]")
    console.print(f"Mean: {mean_lat:.2f} ms")
    console.print(f"p50:  {p50:.2f} ms")
    console.print(f"p95:  {p95:.2f} ms")
    console.print(f"FPS:  {fps:.2f}")

if __name__ == "__main__":
    main()
