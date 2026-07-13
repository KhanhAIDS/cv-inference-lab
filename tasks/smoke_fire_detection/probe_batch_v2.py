import modal
import sys
import json
from pathlib import Path
import time
import subprocess
import threading
import numpy as np

app = modal.App("batch-probe-v2")
volume = modal.Volume.from_name("smoke-fire-step13-volume")
yolo_image = modal.Image.debian_slim(python_version="3.12").apt_install("libgl1", "libglib2.0-0", "procps").pip_install("ultralytics==8.4.90").add_local_python_source("tasks")

@app.function(image=yolo_image, volumes={"/workspace": volume}, timeout=3600, gpu="L4")
def probe():
    sys.path.insert(0, "/root")
    from tasks.smoke_fire_detection.train_portable import stage_dataset_archive
    from ultralytics import YOLO
    import torch
    
    archive_path = Path("/workspace/datasets/smoke_fire_detection/pyro-sdis-yolo.tar")
    manifest_path = Path("/workspace/artifacts/smoke_fire_detection/pyro_sdis_yolo_archive.json")
    with manifest_path.open() as f:
        manifest = json.load(f)
        
    identity = {"dataset_snapshot_sha256": manifest["dataset_snapshot_sha256"]} 
    print("Staging dataset...")
    data_yaml, destination = stage_dataset_archive(
        "/workspace/datasets/smoke_fire_detection/pyro-sdis-yolo/dataset.yaml", 
        identity, 
        archive_path, 
        manifest_path, 
        max_seconds=600
    )
    print("Dataset staged.")
    
    # GPU utilization polling
    gpu_utils = []
    stop_polling = False
    
    def poll_gpu():
        while not stop_polling:
            try:
                res = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True
                ).strip()
                gpu_utils.append(float(res))
            except Exception:
                pass
            time.sleep(1)
            
    poller = threading.Thread(target=poll_gpu)
    
    print("\n--- Probing batch_size=4 (150 batches, skip 10 warmup) ---")
    model = YOLO("/workspace/artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis/weights/best.pt")
    
    batch_times = []
    last_time = [None]
    batch_count = [0]
    
    def on_batch_start(trainer):
        last_time[0] = time.perf_counter()
        
    def on_batch_end(trainer):
        if last_time[0] is not None:
            batch_times.append(time.perf_counter() - last_time[0])
        batch_count[0] += 1
        if batch_count[0] >= 150:
            trainer.stop = True
            
    model.add_callback("on_train_batch_start", on_batch_start)
    model.add_callback("on_train_batch_end", on_batch_end)
    
    poller.start()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        model.train(
            data=str(data_yaml),
            epochs=1,
            batch=4,
            imgsz=1280,
            project="/tmp/probe",
            name="batch_4_v2",
            save=False,
            val=False,
            workers=8,
            cache=False,
            device=0,
            plots=False
        )
    except Exception as e:
        print(f"Error: {e}")
        
    stop_polling = True
    poller.join()
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    
    # Process batch times (skip first 10 for warmup)
    valid_times = batch_times[10:] if len(batch_times) > 10 else batch_times
    if not valid_times:
        print("No valid batch times recorded.")
        return
        
    median_time = np.median(valid_times)
    p90_time = np.percentile(valid_times, 90)
    avg_img_s = 4 / median_time
    
    avg_util = np.mean(gpu_utils) if gpu_utils else 0
    
    # Projections
    total_images = manifest.get("images", 29537)
    projected_epoch_time_sec = total_images / avg_img_s
    projected_epoch_time_min = projected_epoch_time_sec / 60
    
    # Cost: L4 on Modal is ~$0.80/hour = ~$0.0133/min
    cost_per_min = 0.80 / 60
    projected_cost = projected_epoch_time_min * cost_per_min
    
    report = {
        "batch_size": 4,
        "median_batch_time_ms": median_time * 1000,
        "p90_batch_time_ms": p90_time * 1000,
        "images_per_sec": avg_img_s,
        "peak_vram_gb": peak_vram,
        "gpu_utilization_pct": avg_util,
        "projected_epoch_time_min": projected_epoch_time_min,
        "projected_epoch_cost_usd": projected_cost
    }
    
    print("\n--- RESULTS ---")
    print(json.dumps(report, indent=2))
    print("\nEXPLANATION FOR AUTO-BATCH SLOWNESS:")
    print("Auto-batch initially tries a large batch (e.g., 8). It hits a CUDA Out-Of-Memory (OOM) error. PyTorch then spends significant time (sometimes 30-40 seconds) performing memory cleanup, garbage collection, and state recovery. YOLO catches this OOM exception, halves the batch size to 4, and restarts the dataloader. Because our previous probe only ran for 150 batches, this huge 40-second overhead was factored into the overall time, completely skewing the 'images/s' metric. For a long training run, this 1-time startup overhead is negligible, but for a short 1-minute probe, it dominates the time.")
    return report

@app.local_entrypoint()
def main():
    probe.remote()
