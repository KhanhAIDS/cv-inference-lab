import argparse
import math
import tkinter as tk
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageTk

from review_dfire_label_fixes import CLASS_COLORS, CLASS_IDS, CLASS_NAMES, EDGE_TOLERANCE, IMAGE_EXTENSIONS, SPLITS, fit_display_size


CUTOFF_YEAR = 2026


def read_valid_boxes(label_path):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            box = tuple(float(value) for value in parts[1:])
        except (ValueError, IndexError):
            return None
        if len(parts) != 5 or class_id not in CLASS_IDS or not all(math.isfinite(value) for value in box):
            return None
        x, y, width, height = box
        edges = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)
        if width <= 0 or height <= 0 or min(edges) < -EDGE_TOLERANCE or max(edges) > 1 + EDGE_TOLERANCE:
            return None
        boxes.append((class_id, box))
    return boxes or None


def scan_valid_samples(dataset_root, cutoff_year=CUTOFF_YEAR):
    cutoff = datetime(cutoff_year, 1, 1).timestamp()
    records = []
    for split in SPLITS:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        images = {path.stem: path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = images.get(label_path.stem)
            if image_path is None or image_path.stat().st_mtime >= cutoff or label_path.stat().st_mtime >= cutoff:
                continue
            boxes = read_valid_boxes(label_path)
            if boxes is None:
                continue
            records.append({"split": split, "image": image_path, "label": label_path, "boxes": boxes})
    return records


class ValidLabelReviewApp:
    def __init__(self, dataset_root, records):
        self.dataset_root = dataset_root
        self.records = records
        self.photo = None
        self.current_index = None
        self.root = tk.Tk()
        self.root.title(f"D-Fire valid labels before {CUTOFF_YEAR}")
        self.root.geometry("1100x720")
        self.root.resizable(False, False)
        self.listbox = tk.Listbox(self.root, width=42, exportselection=False)
        self.listbox.pack(side="left", fill="y", padx=8, pady=8)
        panel = tk.Frame(self.root)
        panel.pack(side="right", fill="both", expand=True, padx=(0, 8), pady=8)
        self.canvas = tk.Canvas(panel, background="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(panel, anchor="w", justify="left", height=3)
        self.status.pack(fill="x", pady=(6, 0))
        for index, record in enumerate(records, 1):
            classes = "+".join(CLASS_NAMES[class_id] for class_id in sorted({box[0] for box in record["boxes"]}))
            self.listbox.insert("end", f"{index:05d} {record['split']}/{record['image'].name} | {len(record['boxes'])} bbox | {classes}")
        self.listbox.bind("<<ListboxSelect>>", self.show)
        self.canvas.bind("<Configure>", self.show)
        for key, delta in (("<Left>", -1), ("<Up>", -1), ("<Right>", 1), ("<Down>", 1)):
            self.root.bind(key, lambda event, step=delta: self.move(step))
            self.listbox.bind(key, lambda event, step=delta: self.key_move(step))
        self.listbox.selection_set(0)
        self.root.after(50, self.show)

    def key_move(self, delta):
        self.move(delta)
        return "break"

    def move(self, delta):
        current = self.current_index if self.current_index is not None else 0
        target = max(0, min(len(self.records) - 1, current + delta))
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(target)
        self.listbox.see(target)
        self.show()

    def show(self, event=None):
        if not self.listbox.curselection():
            return
        self.current_index = self.listbox.curselection()[0]
        record = self.records[self.current_index]
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
        width, height = image.size
        canvas_width = max(self.canvas.winfo_width(), 200)
        canvas_height = max(self.canvas.winfo_height(), 200)
        display_width, display_height = fit_display_size(width, height, canvas_width - 24, canvas_height - 24)
        shown = image if image.size == (display_width, display_height) else image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(shown)
        origin_x = (canvas_width - display_width) / 2
        origin_y = (canvas_height - display_height) / 2
        self.canvas.delete("all")
        self.canvas.create_image(origin_x, origin_y, anchor="nw", image=self.photo)
        for class_id, (x, y, box_width, box_height) in record["boxes"]:
            x1 = origin_x + (x - box_width / 2) * display_width
            y1 = origin_y + (y - box_height / 2) * display_height
            x2 = origin_x + (x + box_width / 2) * display_width
            y2 = origin_y + (y + box_height / 2) * display_height
            color = CLASS_COLORS[class_id]
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            self.canvas.create_text(x1 + 3, y1 + 3, anchor="nw", text=f"{class_id} {CLASS_NAMES[class_id]}", fill=color)
        fit_scale = min(1.0, (canvas_width - 24) / width, (canvas_height - 24) / height)
        relative = record["image"].relative_to(self.dataset_root)
        self.status.configure(text=f"{self.current_index + 1}/{len(self.records)} | {relative} | {len(record['boxes'])} bbox hợp lệ | mtime ảnh+label < {CUTOFF_YEAR}\ngốc {width}×{height} | raster {display_width}×{display_height} | scale fit {fit_scale:.6f}\nCyan=smoke | cam=fire | chỉ xem | ←/→/↑/↓ duyệt")

    def run(self):
        self.root.mainloop()


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", type=Path, default=repo_root / "datasets/smoke_fire_detection/D-Fire")
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()
    records = scan_valid_samples(args.dataset)
    class_counts = {CLASS_NAMES[class_id]: sum(box[0] == class_id for record in records for box in record["boxes"]) for class_id in sorted(CLASS_IDS)}
    print({"valid_samples_before_2026": len(records), **class_counts})
    if not args.no_gui:
        if not records:
            raise RuntimeError("Không có ảnh trước 2026 với bbox hợp lệ")
        ValidLabelReviewApp(args.dataset, records).run()


if __name__ == "__main__":
    main()
