import modal
import sys
import json
from pathlib import Path
import time

app = modal.App("batch-probe")
volume = modal.Volume.from_name("smoke-fire-step13-volume")
yolo_image = modal.Image.debian_slim(python_version="3.12").apt_install("libgl1", "libglib2.0-0").pip_install("ultralytics==8.4.90").add_local_python_source("tasks")

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
        
    import shutil
    import subprocess
    
    # Restore dataset.yaml
    data_yaml_path = Path("/workspace/datasets/smoke_fire_detection/pyro-sdis-yolo/dataset.yaml")
    if not data_yaml_path.exists():
        data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", str(archive_path), "dataset.yaml", "-O"], stdout=data_yaml_path.open("wb"))
        
    tmp_dest = Path("/tmp") / f"smoke-fire-pyro-sdis-{manifest['dataset_snapshot_sha256'][:16]}"
    if tmp_dest.exists():
        print(f"Removing existing {tmp_dest} to force re-extraction")
        shutil.rmtree(tmp_dest)
        
    identity = {"dataset_snapshot_sha256": manifest["dataset_snapshot_sha256"]} 
    print("Staging dataset...")
    data_yaml, destination = stage_dataset_archive(str(data_yaml_path), identity, archive_path, manifest_path, max_seconds=600)
    print("Dataset staged.")
    
    batches = [1, 2, 4, -1]
    results = []
    
    for b in batches:
        print(f"\n--- Probing batch_size={b} ---")
        model = YOLO("/workspace/artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis/weights/best.pt")
        
        # Add callback for early stop
        def on_batch_end(trainer):
            if trainer.batch_i >= 150:
                trainer.stop = True
                
        model.add_callback("on_train_batch_end", on_batch_end)
        
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            model.train(
                data=str(data_yaml),
                epochs=1,
                batch=b,
                imgsz=1280,
                project="/tmp/probe",
                name=f"batch_{b}",
                save=False,
                val=False,
                workers=8,
                cache=False,
                device=0,
                plots=False
            )
        except Exception as e:
            print(f"Error for batch {b}: {e}")
            
        t1 = time.time()
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        bs = getattr(model.trainer, "batch_size", b) if hasattr(model, "trainer") else b
        if bs == -1: bs = 1 # Fallback if auto-batch failed
        
        total_images = 150 * bs
        duration = t1 - t0
        img_s = total_images / duration
        print(f"Result batch={b}: VRAM={peak_vram:.2f} GB, images/s={img_s:.2f}, eff_batch={bs}, duration={duration:.1f}s")
        results.append({"batch": b, "vram_gb": peak_vram, "img_s": img_s, "eff_batch": bs, "duration": duration})
        
    return results

@app.local_entrypoint()
def main():
    print(probe.remote())
