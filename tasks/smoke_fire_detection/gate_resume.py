import modal
import sys
import json
from pathlib import Path
import time
import shutil
import hashlib

app = modal.App("exact-resume-gate")
volume = modal.Volume.from_name("smoke-fire-step13-volume")
yolo_image = modal.Image.debian_slim(python_version="3.12").apt_install("libgl1", "libglib2.0-0", "procps").pip_install("ultralytics==8.4.90").add_local_python_source("tasks")

@app.function(image=yolo_image, volumes={"/workspace": volume}, timeout=3600, gpu="L4")
def run_invocation(invocation: str, run_dir_str: str, weights_path_str: str, epochs: int, fraction: float, is_resume: bool = False):
    sys.path.insert(0, "/root")
    from tasks.smoke_fire_detection.train_portable import stage_dataset_archive
    from ultralytics import YOLO
    import torch
    
    # Reload volume to ensure we see files written by the caller
    volume.reload()
    
    print(f"\n--- Starting Invocation {invocation} ---")
    
    # 1. Stage dataset in this container
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
        
    b_weights_dir = Path(run_dir_str) / invocation / "weights"
    b_weights_dir.mkdir(parents=True, exist_ok=True)
    
    if is_resume:
        weights_path = Path(weights_path_str)
        if weights_path.exists():
            yolo_expected_path = b_weights_dir / "last.pt"
            shutil.copy2(weights_path, yolo_expected_path)
            weights_path_str = str(yolo_expected_path)
    
    model = YOLO(weights_path_str)
    
    def copy_last_resume(trainer):
        # Called when Trainer saves a checkpoint
        last_pt = trainer.save_dir / "weights" / "last.pt"
        if last_pt.exists():
            last_resume = trainer.save_dir / "weights" / "last_resume.pt"
            shutil.copy2(last_pt, last_resume)
            
            # Print hash
            import hashlib
            with open(last_resume, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            print(f"[{invocation}] Copied last_resume.pt. SHA256: {h}")

    if not is_resume:
        model.add_callback("on_model_save", copy_last_resume)

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=4,
        imgsz=1280,
        project=run_dir_str,
        name=invocation,
        save=True,
        val=False,
        workers=8,
        cache=False,
        device=0,
        plots=False,
        fraction=fraction,
        resume=is_resume,
        exist_ok=True
    )
    
    print(f"--- Finished Invocation {invocation} ---")
    
    volume.commit()
    return True

@app.function(image=yolo_image, volumes={"/workspace": volume}, timeout=3600, gpu="L4")
def run_gate():
    import torch
    from pathlib import Path
    import shutil
    
    base_dir = Path("/workspace/artifacts/smoke_fire_detection/runs/resume_gate")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    fraction = 0.0175
    
    # Invocation A: Train for exactly 1 epoch to get a clean boundary
    run_invocation.remote("A", str(base_dir), "/workspace/artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis/weights/best.pt", 1, fraction)
    volume.reload()
    
    # Check A
    last_A = base_dir / "A" / "weights" / "last_resume.pt"
    if not last_A.exists():
        print("FAIL: Invocation A did not produce last_resume.pt")
        return False
        
    checkpoint_A = torch.load(last_A, map_location="cpu", weights_only=False)
    epoch_A = checkpoint_A.get("epoch", -1)
    opt_A = checkpoint_A.get("optimizer", None)
    updates_A = checkpoint_A.get("updates", -1)
    
    print(f"[Gate] Invocation A Checkpoint: epoch={epoch_A}, updates={updates_A}, optimizer_present={opt_A is not None}")
    
    if epoch_A != 0 or opt_A is None:
        print("FAIL: Invocation A checkpoint invalid.")
        return False
        
    # Modify checkpoint for B to force YOLOv8 to write to B and train up to epoch 2
    checkpoint_A["train_args"]["project"] = str(base_dir)
    checkpoint_A["train_args"]["name"] = "B"
    checkpoint_A["train_args"]["epochs"] = 2
    checkpoint_A["train_args"]["save_dir"] = str(base_dir / "B")
    
    # Save the modified checkpoint into B's directory directly
    b_weights_dir = base_dir / "B" / "weights"
    b_weights_dir.mkdir(parents=True, exist_ok=True)
    last_B_resume = b_weights_dir / "last_resume_modified.pt"
    torch.save(checkpoint_A, last_B_resume)
    volume.commit()
    
    # Invocation B: Resume from the modified checkpoint
    # Target epochs = 2 (it will resume from epoch 1 to 2, training exactly 1 epoch)
    run_invocation.remote("B", str(base_dir), str(last_B_resume), 2, fraction, is_resume=True)
    volume.reload()
    
    # Check B
    last_B = base_dir / "B" / "weights" / "last.pt"
    if not last_B.exists():
        print("FAIL: Invocation B did not produce last.pt")
        return False
        
    checkpoint_B = torch.load(last_B, map_location="cpu", weights_only=False)
    epoch_B = checkpoint_B.get("epoch", -1)
    updates_B = checkpoint_B.get("updates", -1)
    
    print(f"[Gate] Invocation B Checkpoint: epoch={epoch_B}, updates={updates_B}")
    
    if epoch_B != 1:
        print(f"FAIL: Invocation B epoch {epoch_B} != 1.")
        return False
        
    if updates_B <= updates_A:
        print("FAIL: Invocation B updates did not increase.")
        return False
        
    print("SUCCESS: Exact-resume gate passed!")
    return True

@app.local_entrypoint()
def main():
    run_gate.remote()
