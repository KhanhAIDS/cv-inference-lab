import os
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

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

rfdetr_image = modal.Image.debian_slim(python_version="3.12").apt_install(
    "libgl1",
    "libglib2.0-0",
).pip_install(
    "torch==2.12.1+cu130",
    "torchvision==0.27.1+cu130",
    extra_index_url="https://download.pytorch.org/whl/cu130",
).pip_install(
    "rfdetr==1.8.3",
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

utility_image = add_tasks(utility_image)
yolo_image = add_tasks(yolo_image)
rfdetr_image = add_tasks(rfdetr_image)
volume = modal.Volume.from_name("smoke-fire-step13-volume", create_if_missing=True)
PYRONEAR_REVISION = "cd075ce"
PYRONEAR_SHA256 = "2898ecdf96eae513cdca995e4325d3536472016db2131588c7c4e27d5a829483"

def workspace_path(path: str):
    value = Path(path)
    if value.is_absolute():
        return value
    return Path("/workspace") / value

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

@app.function(image=yolo_image, gpu="L4", volumes={"/workspace": volume}, timeout=7200)
def profile_yolo_candidate(
    weights: str,
    images: str,
    candidate_id: str = "",
    class_names: str = "",
    imgsz: int = 1280,
    conf: float = 0.05,
    iou: float = 0.6,
    warmup: int = 30,
    measured: int = 300,
    rounds: int = 3,
    out: str = "artifacts/smoke_fire_detection/profile_report.json",
):
    command = [
        "tasks.smoke_fire_detection.eval",
        "profile",
        "--backend",
        "yolo",
        "--weights",
        str(workspace_path(weights)),
        "--images",
        str(workspace_path(images)),
        "--candidate-id",
        candidate_id,
        "--imgsz",
        str(imgsz),
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--warmup",
        str(warmup),
        "--measured",
        str(measured),
        "--rounds",
        str(rounds),
        "--out",
        str(workspace_path(out)),
    ]
    if class_names:
        command.extend(["--class-names", class_names])
    result = run_module(command)
    volume.commit()
    return result

@app.function(image=rfdetr_image, gpu="L4", volumes={"/workspace": volume}, timeout=7200)
def profile_rfdetr_candidate(
    weights: str,
    images: str,
    candidate_id: str = "",
    class_names: str = "",
    imgsz: int = 1280,
    conf: float = 0.05,
    warmup: int = 30,
    measured: int = 300,
    rounds: int = 3,
    out: str = "artifacts/smoke_fire_detection/profile_report.json",
):
    command = [
        "tasks.smoke_fire_detection.eval",
        "profile",
        "--backend",
        "rfdetr",
        "--weights",
        str(workspace_path(weights)),
        "--images",
        str(workspace_path(images)),
        "--candidate-id",
        candidate_id,
        "--imgsz",
        str(imgsz),
        "--conf",
        str(conf),
        "--warmup",
        str(warmup),
        "--measured",
        str(measured),
        "--rounds",
        str(rounds),
        "--out",
        str(workspace_path(out)),
    ]
    if class_names:
        command.extend(["--class-names", class_names])
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

@app.local_entrypoint()
def pyro_sdis_cli(action: str = "cache"):
    if action == "build-index":
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
    elif action == "profile-yolo":
        print(profile_yolo_candidate.remote())
    elif action == "profile-rfdetr":
        print(profile_rfdetr_candidate.remote())
    else:
        raise ValueError(f"unsupported action: {action}")

@app.function(image=yolo_image, gpu="L4", volumes={"/workspace": volume}, timeout=86400)
def evaluate(
    weights: str = "artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis/weights/best.pt",
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
