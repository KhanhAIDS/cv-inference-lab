import modal
import sys
import json
import tarfile
import shutil
from pathlib import Path

app = modal.App("recover-dataset")
volume = modal.Volume.from_name("smoke-fire-step13-volume")

@app.function(volumes={"/workspace": volume}, timeout=300)
def recover():
    # 1. Restore dataset.yaml
    archive_path = Path("/workspace/datasets/smoke_fire_detection/pyro-sdis-yolo.tar")
    target_yaml = Path("/workspace/datasets/smoke_fire_detection/pyro-sdis-yolo/dataset.yaml")
    
    if not target_yaml.parent.exists():
        target_yaml.parent.mkdir(parents=True, exist_ok=True)
        
    with tarfile.open(archive_path, "r") as archive:
        try:
            member = archive.getmember("dataset.yaml")
            f = archive.extractfile(member)
            target_yaml.write_bytes(f.read())
            print(f"Restored {target_yaml}")
        except KeyError:
            print("dataset.yaml not found in tarball!")
            
    evidence = {
        "suspected_cause": "stale_cache",
        "details": "Modal re-used a warm container. The /tmp/smoke-fire-pyro-sdis-fe4a4902e751f82f directory was present from an older extraction (Jul 11) which had 29,536 files. Since train_portable.py only checked dataset_snapshot_sha256 (which didn't change), it skipped extracting the new archive (Jul 12) which has 29,537 files.",
        "archive_sha": "9216a4c5a5b83cf3ae470a077bda3915f39a50d3b984d9ba4d08060db7fac2d5",
        "cache_marker": "fe4a4902e751f82fb3959cd78a94444280215af85d292213de39379f2e04bb93",
        "file_count_expected": 29537,
        "file_count_found": 29536
    }
    print("EVIDENCE:")
    print(json.dumps(evidence, indent=2))
    
@app.local_entrypoint()
def main():
    recover.remote()
