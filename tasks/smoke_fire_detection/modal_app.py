import os
import subprocess
from pathlib import Path

import modal

app = modal.App("smoke-fire-detection-yolo")
utility_image = modal.Image.debian_slim().pip_install(
    "pyyaml",
    "rich",
)
yolo_image = modal.Image.debian_slim().apt_install(
    "libgl1",
    "libglib2.0-0",
).pip_install(
    "ultralytics",
    "opencv-python-headless",
    "numpy",
    "pyyaml",
    "rich",
)

def add_tasks(image):
    if hasattr(image, "add_local_python_source"):
        return image.add_local_python_source("tasks")
    if hasattr(image, "add_local_dir"):
        return image.add_local_dir("tasks", remote_path="/root/tasks")
    return image

utility_image = add_tasks(utility_image)
yolo_image = add_tasks(yolo_image)
volume = modal.Volume.from_name("smoke-fire-lab-volume", create_if_missing=True)

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
