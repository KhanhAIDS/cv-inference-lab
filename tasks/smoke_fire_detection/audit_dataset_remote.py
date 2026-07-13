import modal
import os
import tarfile

app = modal.App("khanhaids-audit-dataset")
vol = modal.Volume.from_name("smoke-fire-step13-volume")

@app.function(
    volumes={"/__modal/volumes/vo-ijkJsjbbCMufJyHWIy8vjo": vol}
)
def audit_archive():
    archive_path = "/__modal/volumes/vo-ijkJsjbbCMufJyHWIy8vjo/datasets/smoke_fire_detection/pyro-sdis-yolo.tar"
    if not os.path.exists(archive_path):
        print(f"Archive not found: {archive_path}")
        return
    
    with tarfile.open(archive_path, "r") as tar:
        members = tar.getnames()
        
        train_images = [m for m in members if "images/train/" in m and m.endswith(".jpg")]
        print(f"Train images in archive (.jpg): {len(train_images)}")
        
        # Save the list
        out_path = "/__modal/volumes/vo-ijkJsjbbCMufJyHWIy8vjo/artifacts/smoke_fire_detection/runs/train_images_list.txt"
        with open(out_path, "w") as f:
            for m in sorted(train_images):
                f.write(m + "\n")
        print(f"Wrote list to {out_path}")

if __name__ == "__main__":
    with app.run():
        audit_archive.remote()
