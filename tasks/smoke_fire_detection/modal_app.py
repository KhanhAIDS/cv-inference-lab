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

@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=3600)
def prepare_split(
    data_root: str = "datasets/smoke_fire_detection/D-Fire",
    out: str = "artifacts/smoke_fire_detection/split",
    audit_out: str = "artifacts/smoke_fire_detection/dfire_audit.json",
):
    result = run_module([
        "tasks.smoke_fire_detection.dataset",
        "dfire",
        "--data-root",
        str(workspace_path(data_root)),
        "--out",
        str(workspace_path(out)),
        "--audit-out",
        str(workspace_path(audit_out)),
    ])
    volume.commit()
    return result

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
    smoke_only: bool = True,
    smoke_batches: int = 300,
    smoke_use_archive: bool = True,
    batch: int = 4,
    resume: str = "auto",
    max_epochs_per_invocation: int = 1,
    max_runtime_seconds: int = 20400,
    dataset_stage_max_seconds: int = 600,
    subset_images: int = 0,
    free_quota: bool = False,
    paid_spend_usd: float = -1.0,
    paid_cap_usd: float = 20.0,
    paid_stop_threshold_usd: float = 18.0,
    code_commit: str = "unverified",
):
    from tasks.smoke_fire_detection.train import parse_args, run_training

    if code_commit == "unverified":
        raise ValueError("code_commit is required for portable training")
    model_value = str(workspace_path(model)) if Path(model).parent != Path(".") else model
    command = [
        "--data", str(workspace_path(data)),
        "--model", model_value,
        "--run-name", run_name,
        "--project", "/workspace/artifacts/smoke_fire_detection/runs",
        "--dataset-snapshot", "/workspace/artifacts/smoke_fire_detection/pyro_sdis_snapshot.json",
        "--dataset-audit", "/workspace/artifacts/smoke_fire_detection/pyro_sdis_audit.json",
        "--dependency-lock", "/root/requirements.lock",
        "--runtime-lock", "/root/requirements-cu130.lock",
        "--code-commit", code_commit,
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
        "--dataset-archive-manifest", "/workspace/artifacts/smoke_fire_detection/pyro_sdis_yolo_archive.json",
        "--smoke-batches", str(smoke_batches),
        "--subset-images", str(subset_images),
        "--paid-cap-usd", str(paid_cap_usd),
        "--paid-stop-threshold-usd", str(paid_stop_threshold_usd),
    ]
    if smoke_only:
        command.append("--smoke-only")
    if smoke_use_archive:
        command.append("--smoke-use-archive")
    if free_quota:
        command.append("--free-quota")
    if paid_spend_usd >= 0:
        command.extend(["--paid-spend-usd", str(paid_spend_usd)])
    result = run_training(parse_args(command), sync_callback=volume.commit)
    volume.commit()
    return result


@app.function(image=yolo_image, gpu="L4", volumes={"/workspace": volume}, timeout=3600)
def audit_batch4(
    data: str = "datasets/smoke_fire_detection/pyro-sdis-yolo/dataset.yaml",
    model: str = "artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis/weights/best.pt",
    out: str = "artifacts/smoke_fire_detection/yolo26x_batch4_probe_report_v2.json",
    warmup_batches: int = 20,
    measured_batches: int = 100,
    l4_gpu_usd_per_second: float = 0.000222,
    remaining_gpu_budget_usd: float = 9.0,
):
    import math
    import statistics
    import torch
    from tasks.smoke_fire_detection.train import (
        STAGING_COUNTS,
        runtime_metadata,
        prepare_resume_gate_subset,
        stage_dataset_archive,
        verified_dataset_identity,
        write_json_atomic,
    )
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    if warmup_batches != 20 or measured_batches != 100:
        raise ValueError("batch-4 audit requires exactly 20 warm-up and 100 measured batches")
    if l4_gpu_usd_per_second <= 0 or remaining_gpu_budget_usd <= 0:
        raise ValueError("batch-4 audit requires positive price and remaining budget")
    data_path = workspace_path(data)
    model_path = workspace_path(model)
    output_path = workspace_path(out)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    identity_args = SimpleNamespace(
        data=str(data_path),
        dataset_snapshot="/workspace/artifacts/smoke_fire_detection/pyro_sdis_snapshot.json",
        dataset_audit="/workspace/artifacts/smoke_fire_detection/pyro_sdis_audit.json",
        dependency_lock="/root/requirements.lock",
        runtime_lock="/root/requirements-cu130.lock",
    )
    identity = verified_dataset_identity(identity_args)
    data_yaml, staging = stage_dataset_archive(
        data_path,
        identity,
        "/workspace/datasets/smoke_fire_detection/pyro-sdis-yolo.tar",
        "/workspace/artifacts/smoke_fire_detection/pyro_sdis_yolo_archive.json",
        max_seconds=600,
    )
    if staging.get("counts") != {
        **STAGING_COUNTS,
        "archive_files": staging["counts"]["archive_files"],
        "dataset_yaml_sha256": staging["counts"]["dataset_yaml_sha256"],
    }:
        raise ValueError("batch-4 audit strict staging counts invalid")
    data_yaml, subset = prepare_resume_gate_subset(
        data_yaml,
        "/tmp/yolo26x_batch4_probe",
        480,
        20260707,
    )
    augmentation = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.4,
    }
    observed_args = {}
    batch_started = [None]
    batch_durations = []
    total_batches = [0]
    validation_started = [False]
    model_instance = YOLO(str(model_path))

    class Batch4AuditTrainer(DetectionTrainer):
        def validate(self):
            return {}, 0.0

        def final_eval(self):
            return

    def on_train_start(trainer):
        observed_args.update({
            "imgsz": trainer.args.imgsz,
            "batch": trainer.args.batch,
            "amp": trainer.args.amp,
            "workers": trainer.args.workers,
            "seed": trainer.args.seed,
            "val": trainer.args.val,
            "save": trainer.args.save,
            "cache": trainer.args.cache,
            "augmentation": {name: getattr(trainer.args, name) for name in augmentation},
        })

    def on_train_batch_start(trainer):
        batch_started[0] = time.perf_counter()

    def on_train_batch_end(trainer):
        if batch_started[0] is None:
            raise RuntimeError("batch timing start callback missing")
        total_batches[0] += 1
        if total_batches[0] > warmup_batches:
            batch_durations.append(time.perf_counter() - batch_started[0])

    def on_val_start(trainer):
        validation_started[0] = True

    model_instance.add_callback("on_train_start", on_train_start)
    model_instance.add_callback("on_train_batch_start", on_train_batch_start)
    model_instance.add_callback("on_train_batch_end", on_train_batch_end)
    model_instance.add_callback("on_val_start", on_val_start)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_instance.train(
        trainer=Batch4AuditTrainer,
        data=str(data_yaml),
        epochs=1,
        imgsz=1280,
        batch=4,
        patience=0,
        seed=20260707,
        workers=8,
        device="0",
        amp=True,
        save=False,
        val=False,
        cache=False,
        plots=False,
        project="/tmp/yolo26x_batch4_probe",
        name="run",
        exist_ok=True,
        **augmentation,
    )
    if len(batch_durations) != measured_batches:
        raise RuntimeError(f"batch-4 audit measured {len(batch_durations)} batches, expected {measured_batches}")
    if validation_started[0]:
        raise RuntimeError("batch-4 audit validation ran despite val=False")
    if observed_args != {
        "imgsz": 1280,
        "batch": 4,
        "amp": True,
        "workers": 8,
        "seed": 20260707,
        "val": False,
        "save": False,
        "cache": False,
        "augmentation": augmentation,
    }:
        raise ValueError(f"batch-4 audit runtime configuration drift: {observed_args}")
    ordered = sorted(batch_durations)
    median_seconds = statistics.median(batch_durations)
    p90_seconds = ordered[math.ceil(0.9 * len(ordered)) - 1]
    measured_seconds = sum(batch_durations)
    images_per_second = measured_batches * 4 / measured_seconds
    projected_epoch_seconds = STAGING_COUNTS["train_images"] / images_per_second
    projected_epoch_gpu_cost = projected_epoch_seconds * l4_gpu_usd_per_second
    projected_total_gpu_cost = projected_epoch_gpu_cost * 20
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    report = {
        "schema": "smoke-fire-yolo26x-batch4-audit-v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate": {
            "technical_pass": peak_vram_bytes <= int(total_vram_bytes * 0.8),
            "modal_budget_pass": projected_total_gpu_cost <= remaining_gpu_budget_usd,
            "full_training_allowed": peak_vram_bytes <= int(total_vram_bytes * 0.8) and projected_total_gpu_cost <= remaining_gpu_budget_usd,
        },
        "real_data": {
            "dataset_identity": identity,
            "strict_staging": staging,
            "train_images": STAGING_COUNTS["train_images"],
            "audit_subset": subset,
        },
        "configuration": observed_args,
        "measurement": {
            "warmup_batches_excluded": warmup_batches,
            "measured_batches": measured_batches,
            "measured_images": measured_batches * 4,
            "median_batch_seconds": median_seconds,
            "p90_batch_seconds": p90_seconds,
            "throughput_images_per_second": images_per_second,
            "peak_vram_bytes": peak_vram_bytes,
            "total_vram_bytes": total_vram_bytes,
            "vram_utilization_fraction": peak_vram_bytes / total_vram_bytes,
        },
        "projection": {
            "train_images": STAGING_COUNTS["train_images"],
            "epoch_seconds": projected_epoch_seconds,
            "epoch_gpu_cost_usd": projected_epoch_gpu_cost,
            "twenty_epoch_gpu_cost_usd": projected_total_gpu_cost,
            "l4_gpu_usd_per_second": l4_gpu_usd_per_second,
            "remaining_gpu_budget_usd": remaining_gpu_budget_usd,
            "cost_scope": "GPU only; excludes CPU, memory, volume, and startup",
            "price_source": "https://modal.com/pricing",
        },
        "runtime": runtime_metadata(),
        "model": {
            "path": str(model_path),
            "sha256": sha256_path(model_path),
        },
    }
    write_json_atomic(output_path, report)
    volume.commit()
    return report

@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=3600)
def artifact_status(run_name: str = "yolo26x_pyro_sdis"):
    run_dir = Path("/workspace/artifacts/smoke_fire_detection/runs") / run_name
    paths = [
        run_dir / "run_metadata.json",
        run_dir / "smoke_report.json",
        run_dir / "checkpoint_manifest.json",
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


@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=1800, cpu=4, memory=16384)
def verify_strict_staging(
    data: str = "datasets/smoke_fire_detection/pyro-sdis-yolo/dataset.yaml",
    archive: str = "datasets/smoke_fire_detection/pyro-sdis-yolo.tar",
    archive_manifest: str = "artifacts/smoke_fire_detection/pyro_sdis_yolo_archive.json",
    snapshot: str = "artifacts/smoke_fire_detection/pyro_sdis_snapshot.json",
    report_out: str = "artifacts/smoke_fire_detection/pyro_sdis_strict_staging_report.json",
):
    from tasks.smoke_fire_detection.train import (
        read_json,
        sha256_file,
        stage_dataset_archive,
        staging_destination,
        staging_lock_path,
        staging_marker_path,
        staging_temporary_prefix,
        write_json_atomic,
    )

    started = time.perf_counter()
    data_path = workspace_path(data)
    archive_path = workspace_path(archive)
    archive_manifest_path = workspace_path(archive_manifest)
    snapshot_path = workspace_path(snapshot)
    report_path = workspace_path(report_out)
    identity = {"dataset_snapshot_sha256": sha256_file(snapshot_path)}
    manifest = read_json(archive_manifest_path)
    archive_sha256 = manifest["archive_sha256"]
    staging_root = Path("/tmp") / "smoke-fire-strict-staging-audit"
    cache_path = staging_destination(archive_sha256, staging_root)
    fresh_yaml, fresh = stage_dataset_archive(
        data_path,
        identity,
        archive_path,
        archive_manifest_path,
        max_seconds=900,
        staging_root=staging_root,
        force_rebuild=True,
    )
    fresh_marker = read_json(staging_marker_path(cache_path))
    cached_yaml, cached = stage_dataset_archive(
        data_path,
        identity,
        archive_path,
        archive_manifest_path,
        max_seconds=900,
        staging_root=staging_root,
    )
    victim = next((cache_path / "images" / "train").rglob("*.jpg"))
    victim_relative = victim.relative_to(cache_path).as_posix()
    victim.unlink()
    stale_yaml, stale = stage_dataset_archive(
        data_path,
        identity,
        archive_path,
        archive_manifest_path,
        max_seconds=900,
        staging_root=staging_root,
    )
    corrupt_archive = staging_root / "corrupt-archive.tar"
    corrupt_archive.write_bytes(b"not a tar archive")
    corrupt_manifest = dict(manifest)
    corrupt_manifest["archive_sha256"] = sha256_file(corrupt_archive)
    corrupt_manifest_path = staging_root / "corrupt-archive-manifest.json"
    write_json_atomic(corrupt_manifest_path, corrupt_manifest)
    corrupt_cache_path = staging_destination(corrupt_manifest["archive_sha256"], staging_root)
    corrupt_error = None
    try:
        stage_dataset_archive(
            data_path,
            identity,
            corrupt_archive,
            corrupt_manifest_path,
            max_seconds=900,
            staging_root=staging_root,
        )
    except Exception as error:
        corrupt_error = f"{type(error).__name__}: {error}"
    interrupted = cache_path.parent / f"{staging_temporary_prefix(cache_path)}interrupted"
    interrupted.mkdir(parents=True, exist_ok=False)
    (interrupted / "dataset.yaml").write_text("path: interrupted\n", encoding="utf-8")
    interrupted_yaml, interrupted_result = stage_dataset_archive(
        data_path,
        identity,
        archive_path,
        archive_manifest_path,
        max_seconds=900,
        staging_root=staging_root,
    )
    marker_before_lock_test = sha256_file(staging_marker_path(cache_path))
    lock_path = staging_lock_path(cache_path)
    lock_path.mkdir()
    concurrent_error = None
    try:
        stage_dataset_archive(
            data_path,
            identity,
            archive_path,
            archive_manifest_path,
            max_seconds=900,
            staging_root=staging_root,
        )
    except Exception as error:
        concurrent_error = f"{type(error).__name__}: {error}"
    finally:
        lock_path.rmdir()
    marker_after_lock_test = sha256_file(staging_marker_path(cache_path))
    tests = {
        "fresh_extract": {
            "pass": fresh["mode"] == "archive",
            "mode": fresh["mode"],
            "counts": fresh["counts"],
            "dataset_yaml_sha256": fresh["dataset_yaml_sha256"],
        },
        "valid_cache_reuse": {
            "pass": cached["mode"] == "archive_cached" and cached["counts"] == fresh["counts"] and cached_yaml.read_bytes() == fresh_yaml.read_bytes(),
            "mode": cached["mode"],
            "counts_equal": cached["counts"] == fresh["counts"],
            "dataset_yaml_sha256": cached["dataset_yaml_sha256"],
        },
        "stale_cache": {
            "pass": stale["mode"] == "archive" and stale_yaml.is_file() and stale["counts"] == fresh["counts"],
            "deleted_relative_path": victim_relative,
            "mode": stale["mode"],
            "counts": stale["counts"],
        },
        "corrupt_archive": {
            "pass": corrupt_error is not None and not corrupt_cache_path.exists() and not staging_marker_path(corrupt_cache_path).exists(),
            "error": corrupt_error,
            "cache_exists": corrupt_cache_path.exists(),
        },
        "interrupted_extract": {
            "pass": not interrupted.exists() and interrupted_result["mode"] == "archive_cached" and interrupted_yaml.is_file(),
            "mode": interrupted_result["mode"],
            "temporary_exists": interrupted.exists(),
        },
        "concurrent_invocation": {
            "pass": concurrent_error is not None and marker_before_lock_test == marker_after_lock_test,
            "error": concurrent_error,
            "marker_unchanged": marker_before_lock_test == marker_after_lock_test,
        },
    }
    report = {
        "schema": "smoke-fire-pyro-sdis-strict-staging-audit-v1",
        "archive_sha256": archive_sha256,
        "dataset_snapshot_sha256": identity["dataset_snapshot_sha256"],
        "dataset_yaml_sha256": fresh_marker["dataset_yaml_sha256"],
        "exact_counts": fresh_marker["counts"],
        "normalized_empty_labels": fresh["normalized_empty_labels"],
        "cache_path": str(cache_path),
        "tests": tests,
        "all_pass": all(entry["pass"] for entry in tests.values()),
        "suspected_stale_cache": "unproven: Task 228 cache artifact unavailable",
        "staging_bug_eliminated_by_strict_validation": all(entry["pass"] for entry in tests.values()),
        "runtime_cpu_seconds": time.perf_counter() - started,
    }
    write_json_atomic(report_path, report)
    volume.commit()
    if not report["all_pass"]:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


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
    manifest_path = run_dir / "checkpoint_manifest.json"
    result["manifest_exists"] = manifest_path.exists()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result["manifest_completed_epoch"] = manifest.get("completed_epoch")
        result["manifest_stop_reason"] = manifest.get("stop_reason")
    return result


@app.function(image=yolo_image, volumes={"/workspace": volume}, timeout=3600)
def verify_checkpoint_load(run_name: str = "yolo26x_pyro_sdis"):
    import torch
    from ultralytics import YOLO
    from tasks.smoke_fire_detection.train import checkpoint_resume_state

    volume.reload()
    run_dir = Path("/workspace/artifacts/smoke_fire_detection/runs") / run_name
    last_path = run_dir / "weights" / "last_resume.pt"
    manifest_path = run_dir / "checkpoint_manifest.json"
    if not last_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"checkpoint or manifest missing: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = sha256_path(last_path)
    recorded_hash = manifest.get("checkpoint_files", {}).get("last_resume.pt", {}).get("sha256")
    if not recorded_hash or actual_hash != recorded_hash:
        raise ValueError("last.pt checksum mismatch")
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    YOLO(str(last_path))
    state = checkpoint_resume_state(last_path)
    if state["dataset_identity"] != manifest.get("identity"):
        raise ValueError("checkpoint dataset identity mismatch")
    snapshot_path = run_dir / "weights" / f"epoch_{state['epoch_zero_based'] + 1:03d}_resume.pt"
    snapshot_state = checkpoint_resume_state(snapshot_path)
    if snapshot_state["sha256"] != actual_hash:
        raise ValueError("immutable checkpoint snapshot hash mismatch")
    snapshots = {}
    for name, expected in manifest.get("resumable_snapshots", {}).items():
        state_entry = checkpoint_resume_state(run_dir / "weights" / name)
        if state_entry["sha256"] != expected.get("sha256"):
            raise ValueError(f"immutable checkpoint manifest mismatch: {name}")
        snapshots[name] = state_entry
    return {
        "run_dir": str(run_dir),
        "load_ok": True,
        "resume_state_ok": True,
        "checkpoint_epoch_zero_based": state["epoch_zero_based"],
        "next_epoch_one_based": state["next_epoch_one_based"],
        "last_resume_pt_sha256": actual_hash,
        "completed_epoch_manifest": manifest.get("completed_epoch"),
        "updates": state["updates"],
        "optimizer_present": state["optimizer_present"],
        "ema_present": state["ema_present"],
        "scaler_present": state["scaler_present"],
        "scheduler_present": state["scheduler_present"],
        "train_args_present": state["train_args_present"],
        "immutable_snapshot": snapshot_state,
        "resumable_snapshots": snapshots,
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
    smoke_batches: int = 300,
    resume: str = "auto",
    max_epochs_per_invocation: int = 1,
    max_runtime_seconds: int = 20400,
    dataset_stage_max_seconds: int = 600,
    free_quota: bool = False,
    paid_spend_usd: float = -1.0,
    paid_cap_usd: float = 20.0,
    paid_stop_threshold_usd: float = 18.0,
):
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
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
    elif action == "train-smoke":
        print(train_candidate.remote(
            run_name=run_name,
            model=model,
            smoke_only=True,
            batch=batch,
            smoke_batches=smoke_batches,
            resume=resume,
            max_epochs_per_invocation=max_epochs_per_invocation,
            max_runtime_seconds=max_runtime_seconds,
            dataset_stage_max_seconds=dataset_stage_max_seconds,
            free_quota=free_quota,
            paid_spend_usd=paid_spend_usd,
            paid_cap_usd=paid_cap_usd,
            paid_stop_threshold_usd=paid_stop_threshold_usd,
            code_commit=code_commit,
        ))
    elif action == "train":
        print(train_candidate.remote(
            run_name=run_name,
            model=model,
            smoke_only=False,
            batch=batch,
            smoke_batches=smoke_batches,
            resume=resume,
            max_epochs_per_invocation=max_epochs_per_invocation,
            max_runtime_seconds=max_runtime_seconds,
            dataset_stage_max_seconds=dataset_stage_max_seconds,
            free_quota=free_quota,
            paid_spend_usd=paid_spend_usd,
            paid_cap_usd=paid_cap_usd,
            paid_stop_threshold_usd=paid_stop_threshold_usd,
            code_commit=code_commit,
        ))
    elif action == "artifact-status":
        print(artifact_status.remote(run_name))
    elif action == "verify-strict-staging":
        print(verify_strict_staging.remote())
    elif action == "verify-checkpoint":
        print(verify_checkpoint_load.remote(run_name))
    elif action == "audit-checkpoint":
        print(audit_checkpoint.remote(run_name))
    elif action == "verify-patch":
        print(verify_resume_patch.remote())
    elif action == "audit-batch4":
        print(audit_batch4.remote())
    elif action == "resume-gate":
        if paid_spend_usd < 0:
            raise ValueError("resume gate requires current paid spend")
        gate_run_name = run_name if run_name != "yolo26x_pyro_sdis" else "yolo26x_pyro_sdis_resume_gate"
        calibration_model = "artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis/weights/best.pt"
        invocation_a = train_candidate.remote(
            model=calibration_model,
            run_name=gate_run_name,
            smoke_only=False,
            batch=4,
            resume="never",
            max_epochs_per_invocation=1,
            max_runtime_seconds=max_runtime_seconds,
            dataset_stage_max_seconds=dataset_stage_max_seconds,
            subset_images=384,
            paid_spend_usd=paid_spend_usd,
            code_commit=code_commit,
        )
        checkpoint_a = verify_checkpoint_load.remote(gate_run_name)
        invocation_b = train_candidate.remote(
            model=calibration_model,
            run_name=gate_run_name,
            smoke_only=False,
            batch=4,
            resume="always",
            max_epochs_per_invocation=1,
            max_runtime_seconds=max_runtime_seconds,
            dataset_stage_max_seconds=dataset_stage_max_seconds,
            subset_images=384,
            paid_spend_usd=paid_spend_usd,
            code_commit=code_commit,
        )
        checkpoint_b = verify_checkpoint_load.remote(gate_run_name)
        snapshot_a_name = f"epoch_{checkpoint_a['checkpoint_epoch_zero_based'] + 1:03d}_resume.pt"
        snapshot_a = checkpoint_b["resumable_snapshots"].get(snapshot_a_name)
        passed = (
            invocation_b["started_epoch_one_based"] == checkpoint_a["next_epoch_one_based"]
            and checkpoint_b["checkpoint_epoch_zero_based"] == checkpoint_a["checkpoint_epoch_zero_based"] + 1
            and checkpoint_b["updates"] > checkpoint_a["updates"]
            and checkpoint_b["last_resume_pt_sha256"] != checkpoint_a["last_resume_pt_sha256"]
            and snapshot_a is not None
            and snapshot_a["sha256"] == checkpoint_a["immutable_snapshot"]["sha256"]
        )
        result = {
            "schema": "smoke-fire-exact-resume-gate-v1",
            "pass": passed,
            "invocation_a": invocation_a,
            "checkpoint_a": checkpoint_a,
            "invocation_b": invocation_b,
            "checkpoint_b": checkpoint_b,
        }
        if not passed:
            raise RuntimeError(json.dumps(result, sort_keys=True))
        print(result)
    else:
        raise ValueError(f"unsupported action: {action}")

@app.function(image=yolo_image, gpu="L4", volumes={"/workspace": volume}, timeout=86400)
def train(
    data_yaml: str = "artifacts/smoke_fire_detection/split/dataset.yaml",
    model_name: str = "yolo26n.pt",
    run_name: str = "dfire_yolo26n_baseline",
    epochs: int = 100,
    imgsz: int = 640,
    batch: float = 0.95,
    patience: int = 20,
    save_period: int = 1,
    seed: int = 20260707,
    resume: str = "auto",
):
    project_dir = Path("/workspace") / "artifacts" / "smoke_fire_detection" / "runs"
    result = run_module([
        "tasks.smoke_fire_detection.train",
        "--data",
        str(workspace_path(data_yaml)),
        "--model",
        model_name,
        "--run-name",
        run_name,
        "--project",
        str(project_dir),
        "--epochs",
        str(epochs),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--patience",
        str(patience),
        "--save-period",
        str(save_period),
        "--seed",
        str(seed),
        "--resume",
        resume,
    ])
    volume.commit()
    return result

@app.function(image=utility_image, volumes={"/workspace": volume}, timeout=3600)
def checkpoint_status(run_name: str = "dfire_yolo26n_baseline"):
    run_dir = Path("/workspace") / "artifacts" / "smoke_fire_detection" / "runs" / run_name
    weights_dir = run_dir / "weights"
    files = {
        "run_dir": run_dir,
        "weights_dir": weights_dir,
        "last_pt": weights_dir / "last.pt",
        "best_pt": weights_dir / "best.pt",
        "results_csv": run_dir / "results.csv",
        "args_yaml": run_dir / "args.yaml",
    }
    return {name: {"path": str(path), "exists": path.exists()} for name, path in files.items()}

@app.function(image=yolo_image, gpu="L4", volumes={"/workspace": volume}, timeout=86400)
def evaluate(
    weights: str = "artifacts/smoke_fire_detection/runs/dfire_yolo26n_baseline/weights/best.pt",
    data_yaml: str = "artifacts/smoke_fire_detection/split/dataset.yaml",
    out: str = "artifacts/smoke_fire_detection/eval_report.json",
    split: str = "test",
    conf: float = 0.25,
    iou: float = 0.6,
    imgsz: int = 640,
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

@app.function(image=yolo_image, gpu="L4", volumes={"/workspace": volume}, timeout=86400)
def extract_hard_negatives(
    weights: str = "artifacts/smoke_fire_detection/runs/dfire_yolo26n_baseline/weights/best.pt",
    data_yaml: str = "artifacts/smoke_fire_detection/split/dataset.yaml",
    out: str = "artifacts/smoke_fire_detection/hard_negatives.jsonl",
    review_dir: str = "artifacts/smoke_fire_detection/hard_negative_review",
    split: str = "test",
    conf: float = 0.25,
    iou: float = 0.6,
    imgsz: int = 640,
    max_candidates: int = 1000,
    max_review_images: int = 200,
):
    result = run_module([
        "tasks.smoke_fire_detection.eval",
        "hard-negatives",
        "--weights",
        str(workspace_path(weights)),
        "--data",
        str(workspace_path(data_yaml)),
        "--out",
        str(workspace_path(out)),
        "--review-dir",
        str(workspace_path(review_dir)),
        "--split",
        split,
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--imgsz",
        str(imgsz),
        "--max-candidates",
        str(max_candidates),
        "--max-review-images",
        str(max_review_images),
    ])
    volume.commit()
    return result
