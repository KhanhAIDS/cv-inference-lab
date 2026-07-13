import modal

app = modal.App("audit-loader")
volume = modal.Volume.from_name("smoke-fire-step13-volume", create_if_missing=True)
yolo_image = modal.Image.debian_slim(python_version="3.12").apt_install("libgl1", "libglib2.0-0").pip_install("ultralytics==8.4.90")

@app.function(image=yolo_image, volumes={"/workspace": volume}, timeout=3600)
def audit_loader():
    import tarfile
    from pathlib import Path
    import subprocess
    import shutil
    
    import subprocess
    print("Listing /workspace:")
    subprocess.run(["ls", "-la", "/workspace"])
    print("Listing /workspace/artifacts/smoke_fire_detection:")
    subprocess.run(["ls", "-la", "/workspace/artifacts/smoke_fire_detection"])
    print("Listing /workspace/datasets/smoke_fire_detection:")
    subprocess.run(["ls", "-la", "/workspace/datasets/smoke_fire_detection"])
    
    archive_paths = list(Path("/workspace").rglob("*.tar"))
    print(f"Found tar files: {archive_paths}")
    if not archive_paths:
        return "No tar files found"
    archive_path = archive_paths[0]
        
    print("Staging using tarfile filter='data'...")
    destination = Path("/tmp/dataset_filtered")
    destination.mkdir(parents=True, exist_ok=True)
    import tarfile
    with tarfile.open(archive_path, "r") as archive:
        archive.extractall(destination, filter="data")
    
    train_images_dir = destination / "images" / "train"
    actual_images = set(p.name for p in train_images_dir.glob("*.jpg"))
    print(f"stage_dataset_archive extracted {len(actual_images)} train images")
    
    # Extract using standard tar
    Path("/tmp/dataset").mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xf", str(archive_path), "-C", "/tmp/dataset"], check=True)
    raw_images_dir = Path("/tmp/dataset/images/train")
    raw_images = set(p.name for p in raw_images_dir.glob("*.jpg"))
    print(f"tar -xf extracted {len(raw_images)} train images")
    
    missing = raw_images - actual_images
    print(f"Missing in stage_dataset_archive: {missing}")
    
    return list(missing)

@app.local_entrypoint()
def main():
    print(audit_loader.remote())
