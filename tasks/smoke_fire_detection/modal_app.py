import os
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import modal

app = modal.App("smoke-fire-detection-yolo")
utility_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pyarrow==21.0.0",
    "pillow==11.3.0",
    "numpy==2.5.1",
    "pyyaml==6.0.2",
    "rich==14.1.0",
)
yolo_image = modal.Image.debian_slim(python_version="3.12").apt_install(
    "libgl1",
    "libglib2.0-0",
).pip_install(
    "torch==2.12.1+cu130",
    "torchvision==0.27.1+cu130",
    extra_index_url="https://download.pytorch.org/whl/cu130",
).pip_install(
    "ultralytics==8.4.90",
    "opencv-python-headless",
    "numpy==2.5.1",
    "pillow==11.3.0",
    "pyyaml==6.0.2",
    "rich==14.1.0",
)

def add_tasks(image):
    if hasattr(image, "add_local_python_source"):
        return image.add_local_python_source("tasks")
    if hasattr(image, "add_local_dir"):
        return image.add_local_dir("tasks", remote_path="/root/tasks")
    return image

def add_lock(image, filename):
    if hasattr(image, "add_local_file"):
        return image.add_local_file(
            f"tasks/smoke_fire_detection/{filename}",
            remote_path=f"/root/{filename}",
        )
    return image

utility_image = add_tasks(utility_image)
yolo_image = add_lock(add_lock(add_tasks(yolo_image), "requirements.lock"), "requirements-cu130.lock")
volume = modal.Volume.from_name("smoke-fire-step13-volume", create_if_missing=True)
PYRONEAR_REVISION = "cd075ce"
PYRONEAR_SHA256 = "2898ecdf96eae513cdca995e4325d3536472016db2131588c7c4e27d5a829483"
PYRO_ARCHIVE_BYTES = 3_409_295_360
PYRO_ARCHIVE_FILES = 67_273
PYRO_SPLIT_FILES = {"train": 29_537, "val": 4_099}
TRAIN_PROJECT = "/workspace/artifacts/smoke_fire_detection/runs"
TRAIN_RUN_NAME = "yolo26x_pyro_sdis_budget9"
TRAIN_RESUME_CHECKPOINT = f"{TRAIN_PROJECT}/{TRAIN_RUN_NAME}/weights/last_resume.pt"

def workspace_path(path: str):
    value = Path(path)
    if value.is_absolute():
        return value
    return Path("/workspace") / value

def sha256_path(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def run_module(args):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"/root:{env.get('PYTHONPATH', '')}"
    completed = subprocess.run(
        ["python", "-m", *args],
        cwd="/root",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return {
        "command": ["python", "-m", *args],
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }

@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=14400, cpu=4, memory=16384)
def convert_pyro_sdis(
    data_root: str = "datasets/smoke_fire_detection/pyro-sdis",
    out: str = "datasets/smoke_fire_detection/pyro-sdis-yolo",
    audit_out: str = "artifacts/smoke_fire_detection/pyro_sdis_audit.json",
    expected_shards: str = "artifacts/smoke_fire_detection/pyro_sdis_snapshot.json",
):
    result = run_module([
        "tasks.smoke_fire_detection.dataset",
        "pyro-sdis",
        "--data-root",
        str(workspace_path(data_root)),
        "--out",
        str(workspace_path(out)),
        "--audit-out",
        str(workspace_path(audit_out)),
        "--expected-shards",
        str(workspace_path(expected_shards)),
    ])
    volume.commit()
    return result

@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=14400, cpu=4, memory=16384)
def package_pyro_sdis(
    data_root: str = "datasets/smoke_fire_detection/pyro-sdis",
    archive_out: str = "datasets/smoke_fire_detection/pyro-sdis-yolo.tar",
    manifest_out: str = "artifacts/smoke_fire_detection/pyro_sdis_yolo_archive.json",
    snapshot: str = "artifacts/smoke_fire_detection/pyro_sdis_snapshot.json",
):
    from tasks.smoke_fire_detection.dataset import cmd_pyro_sdis, pyro_shard_paths

    source = workspace_path(data_root)
    archive_path = workspace_path(archive_out)
    manifest_path = workspace_path(manifest_out)
    snapshot_path = workspace_path(snapshot)
    if not source.is_dir():
        raise FileNotFoundError(source)
    snapshot_hash = sha256_path(snapshot_path)
    if archive_path.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("dataset_snapshot_sha256") == snapshot_hash and existing.get("archive_sha256") == sha256_path(archive_path):
            return existing
    snapshot_data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    expected_shards = snapshot_data.get("shards", {})
    shards = pyro_shard_paths(source)
    if {path.name for path in shards} != set(expected_shards):
        raise ValueError("raw Pyro-SDIS shard set mismatch")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(f".{archive_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    local_root = Path("/tmp") / f"smoke-fire-package-{snapshot_hash[:16]}"
    shutil.rmtree(local_root, ignore_errors=True)
    local_raw = local_root / "raw" / "data"
    local_yolo = local_root / "yolo"
    local_audit = local_root / "audit.json"
    local_archive = local_root / "pyro-sdis-yolo.tar"
    started = time.perf_counter()
    raw_bytes = 0
    files = 0
    try:
        local_raw.mkdir(parents=True)
        for index, shard in enumerate(shards, start=1):
            target = local_raw / shard.name
            print({"copy_raw_shard": index, "name": shard.name, "bytes": shard.stat().st_size}, flush=True)
            shutil.copy2(shard, target)
            digest = sha256_path(target)
            if digest != expected_shards[shard.name]:
                raise ValueError(f"raw shard hash mismatch: {shard.name}")
            raw_bytes += target.stat().st_size
        cmd_pyro_sdis(SimpleNamespace(
            data_root=str(local_root / "raw"),
            out=str(local_yolo),
            audit_out=str(local_audit),
            expected_shards=str(snapshot_path),
            workers=None,
        ))
        audit = json.loads(local_audit.read_text(encoding="utf-8"))
        if audit.get("invariant_differences"):
            raise ValueError("local Pyro-SDIS conversion invariant mismatch")
        with tarfile.open(local_archive, "w") as archive:
            for path in sorted(local_yolo.rglob("*")):
                if not path.is_file() or path.suffix == ".cache":
                    continue
                archive.add(path, arcname=path.relative_to(local_yolo).as_posix(), recursive=False)
                files += 1
                if files % 1000 == 0:
                    print({"archive_files": files}, flush=True)
        archive_sha256 = sha256_path(local_archive)
        shutil.copy2(local_archive, temporary)
        if sha256_path(temporary) != archive_sha256:
            raise ValueError("persistent archive hash mismatch")
        temporary.replace(archive_path)
        manifest = {
            "schema": "smoke-fire-pyro-sdis-archive-v1",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": str(source),
            "dataset_snapshot_sha256": snapshot_hash,
            "raw_shards": {path.name: expected_shards[path.name] for path in shards},
            "raw_bytes": raw_bytes,
            "files": files,
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_sha256,
            "runtime_seconds": time.perf_counter() - started,
            "provider": "Modal",
            "gpu": None,
            "cost": {"paid_cost_usd": None, "free_quota_usage": "dashboard required"},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        volume.commit()
        return manifest
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(local_root, ignore_errors=True)

@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=7200, cpu=4, memory=16384)
def build_figlib_index(
    data_root: str = "datasets/smoke_fire_detection/FIgLib/FIgLib",
    index_out: str = "artifacts/smoke_fire_detection/figlib_index.jsonl",
    audit_out: str = "artifacts/smoke_fire_detection/figlib_audit.json",
    manifest_out: str = "artifacts/smoke_fire_detection/figlib_split_manifest.json",
    split_mode: str = "camera-disjoint",
    subset_manifest: str = "",
    test_ratio: float = 0.3,
    seed: int = 20260707,
):
    command = [
        "tasks.smoke_fire_detection.dataset",
        "figlib",
        "--data-root",
        str(workspace_path(data_root)),
        "--index-out",
        str(workspace_path(index_out)),
        "--audit-out",
        str(workspace_path(audit_out)),
        "--manifest-out",
        str(workspace_path(manifest_out)),
        "--split-mode",
        split_mode,
        "--val-ratio",
        str(test_ratio),
        "--seed",
        str(seed),
    ]
    if subset_manifest:
        command.extend(["--subset-manifest", str(workspace_path(subset_manifest))])
    result = run_module(command)
    volume.commit()
    return result

@app.function(image=yolo_image, gpu="L4", volumes={"/workspace": volume}, timeout=86400)
def cache_candidate(
    index: str = "artifacts/smoke_fire_detection/figlib_index.jsonl",
    data_root: str = "datasets/smoke_fire_detection/FIgLib/FIgLib",
    weights: str = "artifacts/smoke_fire_detection/runs/dfire_yolo26n_baseline_full_vram/weights/best.pt",
    out: str = "artifacts/smoke_fire_detection/figlib_detector_cache.jsonl",
    split: str = "dev",
    candidate_revision: str = "",
    batch: int = 8,
    imgsz: int = 1280,
    conf: float = 0.05,
    iou: float = 0.6,
    sequence_ids: str = "",
    resume: bool = True,
):
    command = [
        "tasks.smoke_fire_detection.eval",
        "detector-cache",
        "--index",
        str(workspace_path(index)),
        "--data-root",
        str(workspace_path(data_root)),
        "--weights",
        str(workspace_path(weights)),
        "--out",
        str(workspace_path(out)),
        "--split",
        split,
        "--batch",
        str(batch),
        "--imgsz",
        str(imgsz),
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--candidate-revision",
        candidate_revision,
    ]
    if sequence_ids:
        command.extend(["--sequence-ids", sequence_ids])
    if resume:
        command.append("--resume")
    result = run_module(command)
    volume.commit()
    return result

@app.function(image=yolo_image, gpu="L4", volumes={"/workspace": volume}, timeout=86400)
def probe_detector(
    index: str = "artifacts/smoke_fire_detection/figlib_index.jsonl",
    data_root: str = "datasets/smoke_fire_detection/FIgLib/FIgLib",
    weights: str = "artifacts/smoke_fire_detection/runs/dfire_yolo26n_baseline_full_vram/weights/best.pt",
    out: str = "artifacts/smoke_fire_detection/figlib_detector_probe.jsonl",
    candidate_revision: str = "",
    limit: int = 30,
    batch: int = 4,
    imgsz: int = 1280,
):
    command = [
        "tasks.smoke_fire_detection.eval",
        "detector-cache",
        "--index",
        str(workspace_path(index)),
        "--data-root",
        str(workspace_path(data_root)),
        "--weights",
        str(workspace_path(weights)),
        "--out",
        str(workspace_path(out)),
        "--split",
        "dev",
        "--limit",
        str(limit),
        "--batch",
        str(batch),
        "--imgsz",
        str(imgsz),
        "--conf",
        "0.05",
        "--iou",
        "0.6",
        "--candidate-revision",
        candidate_revision,
        "--pilot",
    ]
    result = run_module(command)
    volume.commit()
    return result

@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=3600)
def download_pyronear(
    out: str = "artifacts/smoke_fire_detection/pyronear_yolov8s.pt",
    manifest_out: str = "artifacts/smoke_fire_detection/pyronear_yolov8s_manifest.json",
):
    output_path = workspace_path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/pyronear/yolov8s/resolve/{PYRONEAR_REVISION}/yolov8s.pt?download=true"
    urllib.request.urlretrieve(url, output_path)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    if digest != PYRONEAR_SHA256:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"Pyronear SHA-256 mismatch: expected {PYRONEAR_SHA256}, got {digest}")
    manifest_path = workspace_path(manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "model": "pyronear/yolov8s",
        "revision": PYRONEAR_REVISION,
        "filename": "yolov8s.pt",
        "sha256": digest,
        "url": url,
        "license": "Apache-2.0",
    }, indent=2), encoding="utf-8")
    volume.commit()
    return {"path": str(output_path), "sha256": digest, "revision": PYRONEAR_REVISION}

@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=14400, cpu=4, memory=16384)
def run_temporal(
    command: str = "g0",
    cache: str = "artifacts/smoke_fire_detection/figlib_baseline_dev_cache.jsonl",
    out: str = "artifacts/smoke_fire_detection/temporal_baseline_dev.json",
    events_out: str = "",
    score: str = "smoke",
    bootstrap_samples: int = 1000,
):
    args = [
        "tasks.smoke_fire_detection.temporal_eval",
        command,
        "--cache",
        str(workspace_path(cache)),
        "--out",
        str(workspace_path(out)),
    ]
    if command == "temporal":
        args.extend(["--score", score, "--bootstrap-samples", str(bootstrap_samples)])
        if events_out:
            args.extend(["--events-out", str(workspace_path(events_out))])
    elif command == "diagnose":
        args.extend(["--score", score])
    result = run_module(args)
    volume.commit()
    return result

@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=14400, cpu=4, memory=16384)
def compare_candidates(
    baseline_cache: str = "artifacts/smoke_fire_detection/figlib_baseline_dev_cache.jsonl",
    candidate_cache: str = "artifacts/smoke_fire_detection/figlib_pyronear_dev_cache.jsonl",
    out: str = "artifacts/smoke_fire_detection/checkpoint2_candidate_comparison.json",
    bootstrap_samples: int = 1000,
):
    result = run_module([
        "tasks.smoke_fire_detection.temporal_eval",
        "compare-candidates",
        "--baseline-cache",
        str(workspace_path(baseline_cache)),
        "--candidate-cache",
        str(workspace_path(candidate_cache)),
        "--out",
        str(workspace_path(out)),
        "--bootstrap-samples",
        str(bootstrap_samples),
    ])
    volume.commit()
    return result

@app.function(image=yolo_image, gpu="L4", volumes={"/workspace": volume}, timeout=21600)
def train_candidate(
    data: str = "datasets/smoke_fire_detection/pyro-sdis-yolo/dataset.yaml",
    model: str = "yolo26x.pt",
    run_name: str = "yolo26x_pyro_sdis",
    batch: int = 4,
    resume: str = "auto",
    max_epochs_per_invocation: int = 1,
    max_runtime_seconds: int = 20400,
    dataset_stage_max_seconds: int = 600,
    free_quota: bool = False,
    paid_spend_usd: float = -1.0,
    paid_cap_usd: float = 20.0,
    paid_stop_threshold_usd: float = 18.0,
):
    from tasks.smoke_fire_detection.train import parse_args, run_training

    model_value = str(workspace_path(model)) if Path(model).parent != Path(".") else model
    command = [
        "--data", str(workspace_path(data)),
        "--model", model_value,
        "--run-name", run_name,
        "--project", "/workspace/artifacts/smoke_fire_detection/runs",
        "--provider", "Modal",
        "--runtime-id", "modal-l4",
        "--session-limit-seconds", "21600",
        "--storage-limit", "unknown",
        "--batch", str(batch),
        "--resume", resume,
        "--max-epochs-per-invocation", str(max_epochs_per_invocation),
        "--max-runtime-seconds", str(max_runtime_seconds),
        "--dataset-stage-max-seconds", str(dataset_stage_max_seconds),
        "--dataset-archive", "/workspace/datasets/smoke_fire_detection/pyro-sdis-yolo.tar",
        "--paid-cap-usd", str(paid_cap_usd),
        "--paid-stop-threshold-usd", str(paid_stop_threshold_usd),
    ]
    if free_quota:
        command.append("--free-quota")
    if paid_spend_usd >= 0:
        command.extend(["--paid-spend-usd", str(paid_spend_usd)])
    result = run_training(parse_args(command), sync_callback=volume.commit)
    volume.commit()
    return result


@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=3600)
def artifact_status(run_name: str = "yolo26x_pyro_sdis"):
    run_dir = Path("/workspace/artifacts/smoke_fire_detection/runs") / run_name
    paths = [
        run_dir / "run_metadata.json",
        run_dir / "results.csv",
        run_dir / "args.yaml",
        run_dir / "weights" / "last.pt",
        run_dir / "weights" / "last_resume.pt",
        run_dir / "weights" / "best.pt",
    ]
    return {
        "run_dir": str(run_dir),
        "files": {
            path.name: {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_path(path) if path.exists() else None,
            }
            for path in paths
        },
    }


@app.function(image=yolo_image, gpu="L4", volumes={"/workspace": volume}, timeout=1800)
def verify_resume_patch():
    import shutil
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from ultralytics import YOLO

    test_dir = Path("/tmp/verify_resume_patch")
    shutil.rmtree(test_dir, ignore_errors=True)
    img_dir = test_dir / "images" / "train"
    lbl_dir = test_dir / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    rng = np.random.RandomState(42)
    for i in range(20):
        arr = rng.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"img{i:04d}.jpg")
        if i < 15:
            (lbl_dir / f"img{i:04d}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
        else:
            (lbl_dir / f"img{i:04d}.txt").write_text("")
    data = {"path": str(test_dir), "train": "images/train", "val": "images/train", "names": {0: "smoke"}}
    data_path = test_dir / "dataset.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False))
    run_dir = test_dir / "runs" / "verify_patch"

    def make_on_save(rd, stop_after_n=0):
        counter = {"calls": 0}
        def on_save(trainer):
            resume_path = rd / "weights" / "last_resume.pt"
            tmp = resume_path.with_suffix(".pt.tmp")
            src = rd / "weights" / "last.pt"
            shutil.copy2(src, tmp)
            tmp.replace(resume_path)
            counter["calls"] += 1
            if stop_after_n and counter["calls"] >= stop_after_n:
                trainer.stop = True
        return on_save

    def inspect_checkpoint(path):
        if not path.exists():
            return {"exists": False}
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        is_dict = isinstance(ckpt, dict)
        updates = ckpt.get("updates") if is_dict else None
        
        # Sometimes step is inside optimizer
        opt_step = None
        if is_dict and ckpt.get("optimizer") is not None:
            opt = ckpt["optimizer"]
            if hasattr(opt, "state"):
                for group in opt.state.values():
                    if "step" in group:
                        opt_step = float(group["step"])
                        break

        return {
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
            "top_keys": sorted(ckpt.keys()) if is_dict else [],
            "epoch": ckpt.get("epoch") if is_dict else None,
            "updates": updates,
            "optimizer_step": opt_step,
            "optimizer_present": ckpt.get("optimizer") is not None if is_dict else False,
            "scaler_present": ckpt.get("scaler") is not None if is_dict else False,
            "train_args_present": ckpt.get("train_args") is not None if is_dict else False,
            "ema_present": ckpt.get("ema") is not None if is_dict else False,
        }

    model1 = YOLO("yolo26x.pt")
    model1.add_callback("on_model_save", make_on_save(run_dir, stop_after_n=1))
    model1.train(
        data=str(data_path), epochs=4, imgsz=128, batch=20, patience=0,
        seed=42, workers=0, device="0", amp=True, save_period=1,
        project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
    )
    inv1_resume = inspect_checkpoint(run_dir / "weights" / "last_resume.pt")
    inv1_last = inspect_checkpoint(run_dir / "weights" / "last.pt")
    inv1_best = inspect_checkpoint(run_dir / "weights" / "best.pt")
    result = {
        "invocation_1": {
            "last_resume.pt": inv1_resume,
            "last.pt_stripped": inv1_last,
            "best.pt_stripped": inv1_best,
        }
    }
    if not inv1_resume.get("optimizer_present"):
        shutil.rmtree(test_dir, ignore_errors=True)
        raise ValueError(f"INVOCATION 1 FAILED: last_resume.pt has no optimizer. {result}")
    
    if inv1_resume.get("epoch") is None or inv1_resume.get("epoch") < 0:
        shutil.rmtree(test_dir, ignore_errors=True)
        raise ValueError(f"INVOCATION 1 FAILED: Invalid epoch. {result}")
        
    inv1_epoch = inv1_resume["epoch"]

    resume_path = run_dir / "weights" / "last_resume.pt"
    resume_sha_before = sha256_path(resume_path)
    
    model2 = YOLO(str(resume_path))
    model2.add_callback("on_model_save", make_on_save(run_dir, stop_after_n=1))
    model2.train(resume=str(resume_path), data=str(data_path))
    
    inv2_resume = inspect_checkpoint(run_dir / "weights" / "last_resume.pt")
    inv2_last = inspect_checkpoint(run_dir / "weights" / "last.pt")
    result["invocation_2"] = {
        "resume_source_sha256": resume_sha_before,
        "last_resume.pt": inv2_resume,
        "last.pt_stripped": inv2_last,
    }
    
    inv2_epoch = inv2_resume.get("epoch")
    if inv2_epoch is None or inv2_epoch <= inv1_epoch:
        shutil.rmtree(test_dir, ignore_errors=True)
        raise ValueError(f"INVOCATION 2 FAILED: epoch did not increase. inv1={inv1_epoch}, inv2={inv2_epoch}. {result}")
        
    if not inv2_resume.get("optimizer_present"):
        shutil.rmtree(test_dir, ignore_errors=True)
        raise ValueError(f"INVOCATION 2 FAILED: optimizer lost after resume. {result}")

    if inv2_resume["sha256"] == resume_sha_before:
        shutil.rmtree(test_dir, ignore_errors=True)
        raise ValueError(f"INVOCATION 2 FAILED: weight hash did not change. {result}")
        
    inv1_progress = inv1_resume.get("updates") or inv1_resume.get("optimizer_step") or 0
    inv2_progress = inv2_resume.get("updates") or inv2_resume.get("optimizer_step") or 0
    if inv2_progress <= inv1_progress:
        shutil.rmtree(test_dir, ignore_errors=True)
        raise ValueError(f"INVOCATION 2 FAILED: updates/step did not increase. inv1={inv1_progress}, inv2={inv2_progress}. {result}")

    result["epoch_increased"] = True
    result["inv1_epoch"] = inv1_epoch
    result["inv2_epoch"] = inv2_epoch
    result["inv1_progress"] = inv1_progress
    result["inv2_progress"] = inv2_progress
    shutil.rmtree(test_dir, ignore_errors=True)
    return result


@app.function(image=yolo_image, volumes={"/workspace": volume}, timeout=3600)
def audit_checkpoint(run_name: str = "yolo26x_pyro_sdis"):
    import torch

    run_dir = Path("/workspace/artifacts/smoke_fire_detection/runs") / run_name
    result = {"run_dir": str(run_dir)}
    for name in ("last.pt", "last_resume.pt", "best.pt"):
        path = run_dir / "weights" / name
        entry = {"exists": path.exists()}
        if path.exists():
            entry["bytes"] = path.stat().st_size
            entry["sha256"] = sha256_path(path)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            entry["top_keys"] = sorted(ckpt.keys()) if isinstance(ckpt, dict) else ["not_dict"]
            entry["epoch"] = ckpt.get("epoch") if isinstance(ckpt, dict) else None
            entry["optimizer_present"] = ckpt.get("optimizer") is not None if isinstance(ckpt, dict) else False
            entry["optimizer_type"] = type(ckpt.get("optimizer")).__name__ if isinstance(ckpt, dict) and ckpt.get("optimizer") is not None else None
            entry["train_args_present"] = ckpt.get("train_args") is not None if isinstance(ckpt, dict) else False
            entry["ema_present"] = ckpt.get("ema") is not None if isinstance(ckpt, dict) else False
        result[name] = entry
    return result


@app.function(image=yolo_image, volumes={"/workspace": volume}, timeout=3600)
def verify_checkpoint_load(
    run_name: str = "yolo26x_pyro_sdis",
    data: str = "datasets/smoke_fire_detection/pyro-sdis-yolo/dataset.yaml",
):
    import torch
    from ultralytics import YOLO
    from tasks.smoke_fire_detection.train import checkpoint_resume_state, runtime_resume_trainer, stage_dataset_archive

    volume.reload()
    run_dir = Path("/workspace/artifacts/smoke_fire_detection/runs") / run_name
    last_path = run_dir / "weights" / "last_resume.pt"
    data_path = workspace_path(data)
    if not last_path.exists():
        raise FileNotFoundError(last_path)
    runtime_data_path, staging = stage_dataset_archive(
        data_path,
        "/workspace/datasets/smoke_fire_detection/pyro-sdis-yolo.tar",
        3600,
    )
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    YOLO(str(last_path))
    state = checkpoint_resume_state(last_path)
    trainer = runtime_resume_trainer(runtime_data_path, "/workspace/artifacts/smoke_fire_detection/runs", run_name, device="cpu")(
        overrides={"resume": str(last_path), "data": str(runtime_data_path), "project": "/workspace/artifacts/smoke_fire_detection/runs", "name": run_name, "device": "cpu"}
    )
    runtime_paths = {"data": str(trainer.args.data), "project": str(trainer.args.project), "name": str(trainer.args.name), "save_dir": str(trainer.save_dir)}
    if any("/__modal/volumes/" in value for value in runtime_paths.values()):
        raise ValueError(f"old Volume path remains after rebind: {runtime_paths}")
    if runtime_paths["save_dir"] != f"/workspace/artifacts/smoke_fire_detection/runs/{run_name}":
        raise ValueError(f"runtime save_dir mismatch: {runtime_paths}")
    return {
        "run_dir": str(run_dir),
        "load_ok": True,
        "resume_state_ok": True,
        "checkpoint_epoch_zero_based": state["epoch_zero_based"],
        "next_epoch_one_based": state["next_epoch_one_based"],
        "updates": state["updates"],
        "optimizer_present": state["optimizer_present"],
        "ema_present": state["ema_present"],
        "scaler_present": state["scaler_present"],
        "scheduler_present": state["scheduler_present"],
        "train_args_present": state["train_args_present"],
        "dataset_stage": staging,
        "runtime_paths": runtime_paths,
    }

@app.function(image=utility_image, timeout=3600, cpu=1)
def test_pyro_sdis_converter():
    return run_module(["unittest", "tasks.smoke_fire_detection.test_pyro_sdis_converter"])

@app.local_entrypoint()
def pyro_sdis_cli(
    action: str = "test",
    data_root: str = "datasets/smoke_fire_detection/pyro-sdis",
    out: str = "datasets/smoke_fire_detection/pyro-sdis-yolo",
    audit_out: str = "artifacts/smoke_fire_detection/pyro_sdis_audit.json",
    expected_shards: str = "artifacts/smoke_fire_detection/pyro_sdis_snapshot.json",
    run_name: str = "yolo26x_pyro_sdis",
    model: str = "yolo26x.pt",
    batch: int = 4,
    resume: str = "auto",
    max_epochs_per_invocation: int = 1,
    max_runtime_seconds: int = 20400,
    dataset_stage_max_seconds: int = 600,
    free_quota: bool = False,
    paid_spend_usd: float = -1.0,
    paid_cap_usd: float = 20.0,
    paid_stop_threshold_usd: float = 18.0,
    chunk_dir: str = "datasets/smoke_fire_detection/pyro-sdis-yolo.chunks",
    archive_out: str = "datasets/smoke_fire_detection/pyro-sdis-yolo.tar",
):
    if action == "test":
        print(test_pyro_sdis_converter.remote())
    elif action == "convert":
        print(convert_pyro_sdis.remote(data_root, out, audit_out, expected_shards))
    elif action == "package":
        print(package_pyro_sdis.remote())
    elif action == "build-index":
        print(build_figlib_index.remote())
    elif action == "probe":
        print(probe_detector.remote())
    elif action == "cache":
        print(cache_candidate.remote())
    elif action == "download-pyronear":
        print(download_pyronear.remote())
    elif action == "temporal":
        print(run_temporal.remote())
    elif action == "compare":
        print(compare_candidates.remote())
    elif action == "train":
        print(train_candidate.remote(
            run_name=run_name,
            model=model,
            batch=batch,
            resume=resume,
            max_epochs_per_invocation=max_epochs_per_invocation,
            max_runtime_seconds=max_runtime_seconds,
            dataset_stage_max_seconds=dataset_stage_max_seconds,
            free_quota=free_quota,
            paid_spend_usd=paid_spend_usd,
            paid_cap_usd=paid_cap_usd,
            paid_stop_threshold_usd=paid_stop_threshold_usd,
        ))
    elif action == "artifact-status":
        print(artifact_status.remote(run_name))
    elif action == "verify-checkpoint":
        print(verify_checkpoint_load.remote(run_name))
    elif action == "assemble-dataset":
        print(prepare_dataset_volume.remote(chunk_dir, archive_out))
    elif action == "inspect-dataset-archive":
        print(inspect_dataset_archive.remote())
    elif action == "chunk-status":
        print(volume_chunk_status.remote())
    elif action == "audit-checkpoint":
        print(audit_checkpoint.remote(run_name))
    elif action == "verify-patch":
        print(verify_resume_patch.remote())
    else:
        raise ValueError(f"unsupported action: {action}")

def inspect_pyro_archive(archive_path):
    expected = {
        "images/train": PYRO_SPLIT_FILES["train"],
        "images/val": PYRO_SPLIT_FILES["val"],
        "labels/train": PYRO_SPLIT_FILES["train"],
        "labels/val": PYRO_SPLIT_FILES["val"],
    }
    counts = {key: 0 for key in expected}
    dataset_yaml = False
    with tarfile.open(archive_path, "r") as handle:
        members = [member for member in handle.getmembers() if member.isfile()]
    for member in members:
        if member.name == "dataset.yaml":
            dataset_yaml = True
        for prefix in counts:
            if member.name.startswith(f"{prefix}/"):
                counts[prefix] += 1
    if archive_path.stat().st_size != PYRO_ARCHIVE_BYTES:
        raise ValueError(f"archive bytes mismatch: {archive_path.stat().st_size}")
    if len(members) != PYRO_ARCHIVE_FILES:
        raise ValueError(f"archive file count mismatch: {len(members)}")
    if not dataset_yaml or counts != expected:
        raise ValueError(f"archive layout mismatch: dataset_yaml={dataset_yaml}, counts={counts}")
    return {"bytes": archive_path.stat().st_size, "files": len(members), "counts": counts}


@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=7200, cpu=4, memory=16384)
def prepare_dataset_volume(
    chunk_dir: str = "datasets/smoke_fire_detection/pyro-sdis-yolo.chunks",
    archive_out: str = "datasets/smoke_fire_detection/pyro-sdis-yolo.tar",
):
    volume.reload()
    source_dir = workspace_path(chunk_dir)
    archive_path = workspace_path(archive_out)
    chunks = sorted(path for path in source_dir.glob("*.part") if path.is_file())
    if not chunks:
        raise FileNotFoundError(source_dir)
    names = [path.name for path in chunks]
    expected_names = [f"pyro-sdis-yolo.{index:05d}.part" for index in range(len(chunks))]
    if names != expected_names:
        raise ValueError(f"chunk sequence mismatch: {names}")
    chunk_bytes = sum(path.stat().st_size for path in chunks)
    if chunk_bytes != PYRO_ARCHIVE_BYTES:
        raise ValueError(f"chunk bytes mismatch: {chunk_bytes}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(f".{archive_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as target:
        for chunk in chunks:
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    temporary.replace(archive_path)
    archive = inspect_pyro_archive(archive_path)
    destination = archive_path.with_suffix("")
    staging = destination.with_name(f".{destination.name}.staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    with tarfile.open(archive_path, "r") as handle:
        handle.extractall(staging, filter="data")
    staged_archive = inspect_pyro_archive(archive_path)
    import yaml
    dataset_yaml = staging / "dataset.yaml"
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("dataset.yaml is not a mapping")
    data["path"] = str(destination)
    dataset_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    if destination.exists():
        shutil.rmtree(destination)
    staging.replace(destination)
    volume.commit()
    for chunk in chunks:
        chunk.unlink()
    source_dir.rmdir()
    volume.commit()
    return {
        "archive": str(archive_path),
        "archive_check": archive,
        "staged_archive_check": staged_archive,
        "dataset": str(destination),
        "chunks": len(chunks),
    }


@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=3600, cpu=1)
def inspect_dataset_archive(archive: str = "datasets/smoke_fire_detection/pyro-sdis-yolo.tar"):
    volume.reload()
    archive_path = workspace_path(archive)
    result = inspect_pyro_archive(archive_path)
    return {
        "archive": str(archive_path),
        **result,
    }


@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=3600, cpu=1)
def volume_chunk_status(chunk_dir: str = "datasets/smoke_fire_detection/pyro-sdis-yolo.chunks"):
    volume.reload()
    source_dir = workspace_path(chunk_dir)
    return {
        path.name: path.stat().st_size
        for path in sorted(source_dir.glob("*.part"))
        if path.is_file()
    }


def segment_train_args(run_name, resume_checkpoint, max_epochs, max_runtime_seconds, free_quota, paid_spend_usd, paid_cap_usd, paid_stop_threshold_usd):
    args = [
        "--data", "/workspace/datasets/smoke_fire_detection/pyro-sdis-yolo/dataset.yaml",
        "--model", str(resume_checkpoint),
        "--run-name", run_name,
        "--project", TRAIN_PROJECT,
        "--provider", "Modal",
        "--runtime-id", "modal-l4",
        "--session-limit-seconds", "86400",
        "--storage-limit", "unknown",
        "--batch", "4",
        "--resume", "always",
        "--max-epochs-per-invocation", str(max_epochs),
        "--max-runtime-seconds", str(max_runtime_seconds),
        "--dataset-stage-max-seconds", "7200",
        "--dataset-archive", "/workspace/datasets/smoke_fire_detection/pyro-sdis-yolo.tar",
        "--paid-cap-usd", str(paid_cap_usd),
        "--paid-stop-threshold-usd", str(paid_stop_threshold_usd),
    ]
    if free_quota:
        args.append("--free-quota")
    elif paid_spend_usd >= 0:
        args.extend(["--paid-spend-usd", str(paid_spend_usd)])
    return args


def segment_preflight(run_dir, resume_checkpoint, expected_next_epoch):
    from tasks.smoke_fire_detection.train import checkpoint_resume_state, stage_dataset_archive

    if not 3 <= expected_next_epoch <= 20:
        raise ValueError(f"segment expected next epoch invalid: {expected_next_epoch}")
    volume_root = Path("/workspace").resolve()
    try:
        run_dir.resolve().relative_to(volume_root)
    except ValueError as error:
        raise ValueError(f"segment output is outside mounted Volume: {run_dir}") from error
    state = checkpoint_resume_state(resume_checkpoint)
    if state["next_epoch_one_based"] != expected_next_epoch:
        raise ValueError(f"segment checkpoint epoch mismatch: {state}")
    data_path = workspace_path("datasets/smoke_fire_detection/pyro-sdis-yolo/dataset.yaml")
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    runtime_data_path, staging = stage_dataset_archive(
        data_path,
        workspace_path("datasets/smoke_fire_detection/pyro-sdis-yolo.tar"),
        7200,
    )
    if not runtime_data_path.is_file() or not runtime_data_path.read_text(encoding="utf-8").strip():
        raise ValueError(f"staged dataset unreadable: {runtime_data_path}")
    return {
        "checkpoint": state,
        "dataset_stage": staging,
        "output_path": str(run_dir),
        "runtime_data_path": str(runtime_data_path),
    }


def segment_lock_path(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / "train_segment.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    return lock_path


@app.function(image=yolo_image, gpu="L4", volumes={"/workspace": volume}, timeout=86400)
def train_segment(
    run_name: str = TRAIN_RUN_NAME,
    resume_checkpoint: str = TRAIN_RESUME_CHECKPOINT,
    max_epochs: int = 6,
    max_runtime_seconds: int = 82800,
    free_quota: bool = False,
    paid_spend_usd: float = -1.0,
    paid_cap_usd: float = 20.0,
    paid_stop_threshold_usd: float = 18.0,
    expected_next_epoch: int = 3,
):
    from tasks.smoke_fire_detection.train import checkpoint_resume_state, parse_args, run_training

    if not 1 <= max_epochs <= 8 or not 0 < max_runtime_seconds < 86400:
        raise ValueError("segment bounds invalid")
    if run_name != TRAIN_RUN_NAME:
        raise ValueError(f"segment target run must be {TRAIN_RUN_NAME}")
    volume.reload()
    run_dir = Path(TRAIN_PROJECT) / run_name
    resume_path = workspace_path(resume_checkpoint)
    if resume_path != run_dir / "weights" / "last_resume.pt":
        raise ValueError(f"segment resume checkpoint mismatch: {resume_path}")
    lock_path = segment_lock_path(run_dir)
    try:
        lock_path.mkdir()
    except FileExistsError:
        return {"status": "locked", "run_name": run_name}
    should_chain = False
    result = None
    try:
        preflight = segment_preflight(run_dir, resume_path, expected_next_epoch)
        before = preflight["checkpoint"]
        if before["completed_epoch_one_based"] >= 20:
            return {"status": "complete", "run_name": run_name, "checkpoint": before}
        result = run_training(parse_args(segment_train_args(
            run_name,
            resume_path,
            max_epochs,
            max_runtime_seconds,
            free_quota,
            paid_spend_usd,
            paid_cap_usd,
            paid_stop_threshold_usd,
        )), sync_callback=volume.commit)
        after = checkpoint_resume_state(resume_path)
        if after["completed_epoch_one_based"] <= before["completed_epoch_one_based"]:
            raise RuntimeError(f"segment made no progress: before={before}, after={after}")
        should_chain = after["completed_epoch_one_based"] < 20
        state = {
            "run_name": run_name,
            "preflight": preflight,
            "result": result,
            "checkpoint": after,
            "next_segment_scheduled": should_chain,
        }
        (run_dir / "train_segment_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        volume.commit()
    finally:
        lock_path.rmdir()
    if should_chain:
        call = train_segment.spawn(
            run_name=run_name,
            resume_checkpoint=str(resume_path),
            max_epochs=max_epochs,
            max_runtime_seconds=max_runtime_seconds,
            free_quota=free_quota,
            paid_spend_usd=paid_spend_usd,
            paid_cap_usd=paid_cap_usd,
            paid_stop_threshold_usd=paid_stop_threshold_usd,
            expected_next_epoch=after["next_epoch_one_based"],
        )
        result["next_function_call_id"] = getattr(call, "object_id", str(call))
    return result

@app.function(image=yolo_image, gpu="L4", volumes={"/workspace": volume}, timeout=86400)
def evaluate(
    weights: str = "artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis_budget9/weights/best.pt",
    data_yaml: str = "datasets/smoke_fire_detection/pyro-sdis-yolo/dataset.yaml",
    out: str = "artifacts/smoke_fire_detection/eval_report.json",
    split: str = "val",
    conf: float = 0.25,
    iou: float = 0.6,
    imgsz: int = 1280,
    warmup: int = 50,
    measured: int = 500,
):
    result = run_module([
        "tasks.smoke_fire_detection.eval",
        "accuracy",
        "--weights",
        str(workspace_path(weights)),
        "--data",
        str(workspace_path(data_yaml)),
        "--out",
        str(workspace_path(out)),
        "--split",
        split,
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--imgsz",
        str(imgsz),
        "--warmup",
        str(warmup),
        "--measured",
        str(measured),
    ])
    volume.commit()
    return result
