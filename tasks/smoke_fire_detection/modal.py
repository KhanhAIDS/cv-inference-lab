from pathlib import Path
import modal

app = modal.App("smoke-fire-detection-yolo")
image = modal.Image.debian_slim().pip_install(
    "ultralytics",
    "opencv-python-headless",
    "pandas",
    "pyyaml",
    "rich",
)
volume = modal.Volume.from_name("smoke-fire-lab-volume", create_if_missing=True)

@app.function(image=image, gpu=modal.gpu.L4(count=1), volumes={"/workspace": volume}, timeout=86400)
def train(
    data_yaml: str = "artifacts/smoke_fire_detection/split/dataset.yaml",
    model_name: str = "yolo26n.pt",
    run_name: str = "yolo26n_baseline",
):
    from ultralytics import YOLO

    workspace = Path("/workspace")
    data_path = workspace / data_yaml
    project_dir = workspace / "artifacts" / "smoke_fire_detection" / "runs"
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    model = YOLO(model_name)
    model.train(
        data=str(data_path),
        epochs=100,
        patience=20,
        batch=-1,
        imgsz=640,
        seed=20260707,
        pretrained=True,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )
    volume.commit()
    return str(project_dir / run_name)
