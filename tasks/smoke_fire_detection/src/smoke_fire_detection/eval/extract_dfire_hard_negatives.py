import argparse
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from rich.console import Console
import cv2

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Extract hard negative crops from test negatives")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data-root", type=str, required=True, help="Path to D-Fire test images")
    parser.add_argument("--out-csv", type=str, default="artifacts/dfire/hard_negative_candidates.csv")
    parser.add_argument("--out-crops", type=str, default="artifacts/dfire/hard_negative_crops")
    parser.add_argument("--top-k", type=int, default=100, help="Number of top hard negatives to extract")
    return parser.parse_args()

def is_negative(label_path: Path):
    if not label_path.exists():
        return True
    with open(label_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return len(lines) == 0

def main():
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    test_dir = Path(args.data_root).resolve() / "test"
    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"
    out_csv = Path(args.out_csv).resolve()
    out_crops = Path(args.out_crops).resolve()
    
    if not weights_path.exists() or not images_dir.exists():
        console.print("[red]Weights or test images not found[/red]")
        return
        
    out_crops.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(weights_path)
    
    image_paths = list(images_dir.glob("*.*"))
    negative_images = [img for img in image_paths if is_negative(labels_dir / f"{img.stem}.txt")]
    
    console.print(f"[green]Found {len(negative_images)} negative images in test set. Running inference...[/green]")
    
    hard_negatives = []
    
    # Process in batches or one by one
    for img_path in negative_images:
        results = model(str(img_path), verbose=False)
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    
                    hard_negatives.append({
                        "image_path": str(img_path),
                        "predicted_class": model.names[cls_id],
                        "confidence": conf,
                        "xmin": int(xyxy[0]),
                        "ymin": int(xyxy[1]),
                        "xmax": int(xyxy[2]),
                        "ymax": int(xyxy[3])
                    })
                    
    # Sort by confidence descending
    hard_negatives.sort(key=lambda x: x["confidence"], reverse=True)
    top_k = hard_negatives[:args.top_k]
    
    console.print(f"[green]Found {len(hard_negatives)} false positive detections. Extracting top {len(top_k)} crops...[/green]")
    
    # Extract crops
    records = []
    for idx, item in enumerate(top_k):
        img = cv2.imread(item["image_path"])
        crop = img[item["ymin"]:item["ymax"], item["xmin"]:item["xmax"]]
        crop_name = f"fp_{idx:03d}_{Path(item['image_path']).name}"
        crop_path = out_crops / crop_name
        
        if crop.size > 0:
            cv2.imwrite(str(crop_path), crop)
            
        item["crop_path"] = str(crop_path)
        records.append(item)
        
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    
    console.print(f"[green]Saved hard negative report to {out_csv}[/green]")

if __name__ == "__main__":
    main()
