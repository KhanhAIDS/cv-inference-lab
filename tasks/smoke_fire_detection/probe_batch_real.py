import modal
import sys
import json
from pathlib import Path
import time
import subprocess
import threading
import torch

app = modal.App("batch-probe-real")
volume = modal.Volume.from_name("smoke-fire-step13-volume")
yolo_image = modal.Image.debian_slim(python_version="3.12").apt_install("libgl1", "libglib2.0-0", "procps").pip_install("ultralytics==8.4.90").add_local_python_source("tasks")

@app.function(image=yolo_image, volumes={"/workspace": volume}, timeout=3600, gpu="L4")
def probe_all():
    sys.path.insert(0, "/root")
    from tasks.smoke_fire_detection.train_portable import stage_dataset_archive
    from ultralytics import YOLO
    
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
    
    results = {}
    
    # Auto-batch is tested using -1
    for batch_size in [1, 2, 4, -1]:
        print(f"\n--- Probing batch_size={batch_size} (100 batches, skip 10 warmup) ---")
        
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
        model = YOLO("yolo26x.pt") # Use offical model to prevent resuming state
        
        batch_times = []
        last_time = [None]
        batch_count = [0]
        
        def on_batch_start(trainer):
            last_time[0] = time.perf_counter()
            
        def on_batch_end(trainer):
            if last_time[0] is not None:
                batch_times.append(time.perf_counter() - last_time[0])
            batch_count[0] += 1
            if batch_count[0] >= 110: # 10 warmup + 100 actual
                trainer.stop = True
                
        model.add_callback("on_train_batch_start", on_batch_start)
        model.add_callback("on_train_batch_end", on_batch_end)
        
        poller.start()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            model.train(
                data=str(data_yaml),
                epochs=1,
                batch=batch_size,
                imgsz=1280,
                project="/tmp/probe_real",
                name=f"batch_{batch_size}",
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
        valid_times = batch_times[10:] if len(batch_times) > 10 else batch_times
        avg_batch_time = sum(valid_times) / len(valid_times) if valid_times else 0
        
        # Calculate actual batch size in case of auto-batch
        actual_batch = batch_size
        if batch_size == -1:
            # Need to get it from trainer or estimate it.
            # Assuming trainer.batch_size works but we don't have access to trainer outside. 
            pass 
            
        imgs_per_sec = (batch_size if batch_size != -1 else 0) / avg_batch_time if avg_batch_time > 0 and batch_size != -1 else 0
        avg_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
        
        print(f"Results for batch {batch_size}:")
        print(f"Peak VRAM: {peak_vram:.2f} GB")
        print(f"Avg batch time: {avg_batch_time:.3f} s")
        if batch_size != -1:
            print(f"Throughput: {imgs_per_sec:.2f} imgs/s")
            projected_epoch = (29537 / imgs_per_sec) / 3600 if imgs_per_sec > 0 else 0
            print(f"Projected epoch time: {projected_epoch:.2f} hours")
        print(f"Avg GPU Util: {avg_gpu:.1f}%")
        
        results[str(batch_size)] = {
            "peak_vram_gb": peak_vram,
            "avg_batch_time_s": avg_batch_time,
            "throughput_imgs_s": imgs_per_sec,
            "avg_gpu_util": avg_gpu
        }
        
    print("\n--- FINAL SUMMARY ---")
    print(json.dumps(results, indent=2))
    
    with open("/workspace/artifacts/smoke_fire_detection/batch_probe_real_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    with app.run():
        probe_all.remote()
