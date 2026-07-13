import modal
from tasks.smoke_fire_detection.modal_app import train_candidate, app

@app.local_entrypoint()
def main():
    print("Launching train_candidate on Modal...")
    result = train_candidate.remote(
        data="datasets/smoke_fire_detection/pyro-sdis-yolo/dataset.yaml",
        model="/workspace/artifacts/smoke_fire_detection/runs/yolo26x_pyro_sdis/weights/best.pt",
        run_name="yolo26x_pyro_sdis_warmstart",
        smoke_only=False,
        smoke_use_archive=False,
        batch=4,
        resume="never",
        max_epochs_per_invocation=2,
        max_runtime_seconds=20400,
        paid_spend_usd=5.86,
        code_commit="cdd11611af7d807e4112dd9bcd3d9d9a5daebe0a"
    )
    print("Result:", result)
