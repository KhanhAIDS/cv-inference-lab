from pathlib import Path
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("dfire-yolo-baseline")

    # Define the image with necessary dependencies
    image = modal.Image.debian_slim().pip_install(
        "ultralytics",
        "opencv-python-headless",
        "pandas",
        "pyyaml",
        "tqdm",
        "rich"
    )

    # Use a volume for datasets and artifacts to persist across runs
    volume = modal.Volume.from_name("cv-inference-lab-volume", create_if_missing=True)

    @app.function(
        image=image,
        gpu=modal.gpu.L4(count=1),
        volumes={"/workspace": volume},
        timeout=86400  # 24 hours
    )
    def train(data_yaml: str = "dfire.yaml", model_version: str = "yolov8n.pt", run_name: str = "dfire_yolo_baseline"):
        from ultralytics import YOLO
        
        # Paths inside the modal container
        workspace_dir = Path("/workspace")
        data_path = workspace_dir / "data" / "dfire" / data_yaml
        project_dir = workspace_dir / "artifacts" / "runs"
        
        if not data_path.exists():
            print(f"Data YAML not found at {data_path}. Please make sure dataset is uploaded to the volume.")
            return
            
        print(f"Loading {model_version}...")
        model = YOLO(model_version)
        
        print(f"Starting training run {run_name} on Modal...")
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
            exist_ok=True
        )
        print(f"Training completed. Results are in the modal volume at {project_dir / run_name}")
else:
    # Dummy fallback if modal is not installed
    class DummyApp:
        pass
    app = DummyApp()
