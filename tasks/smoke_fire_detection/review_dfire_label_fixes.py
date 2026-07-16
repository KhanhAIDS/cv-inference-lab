import argparse
import json
import math
import os
import random
import tkinter as tk
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageTk


SPLITS = ("train", "valid", "test")
EXPECTED_COUNTS = {"train": 15500, "valid": 1721, "test": 4306}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
CLASS_IDS = {0, 1}
CLASS_NAMES = {0: "smoke", 1: "fire"}
CLASS_COLORS = {0: "#00d8ff", 1: "#ff8c00"}
EDGE_TOLERANCE = 1e-9
SPLIT_SEED = 20260707
CURRENT_YEAR = datetime.now().year


def repair_box(box):
    x_center, y_center, box_width, box_height = box
    edges = (x_center - box_width / 2, y_center - box_height / 2, x_center + box_width / 2, y_center + box_height / 2)
    clipped = tuple(max(0.0, min(1.0, value)) for value in edges)
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return "drop_zero_area", None
    fixed = ((clipped[0] + clipped[2]) / 2, (clipped[1] + clipped[3]) / 2, clipped[2] - clipped[0], clipped[3] - clipped[1])
    changed = any(abs(before - after) > EDGE_TOLERANCE for before, after in zip(edges, clipped))
    return ("clip_edges", fixed) if changed else ("keep", box)


def repair_label(label_path, source_text=None):
    cleaned_lines = []
    issues = []
    source_text = label_path.read_text(encoding="utf-8") if source_text is None else source_text
    for line_number, line in enumerate(source_text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            box = tuple(float(value) for value in parts[1:])
            valid = len(parts) == 5 and class_id in CLASS_IDS and all(math.isfinite(value) for value in box)
        except (ValueError, IndexError):
            valid = False
        if not valid:
            raise ValueError(f"Label sai cú pháp/class/non-finite: {label_path}:{line_number}")
        action, fixed = repair_box(box)
        if action != "keep":
            issues.append({"line": line_number, "action": action, "class_id": class_id, "original": box, "fixed": fixed})
        if fixed is not None:
            cleaned_lines.append(f"{class_id} " + " ".join(f"{value:.12g}" for value in fixed) if action == "clip_edges" else line.strip())
    cleaned_text = "\n".join(cleaned_lines) + ("\n" if cleaned_lines else "")
    return issues, cleaned_text


def scan_dataset(dataset_root, modified_year=None):
    records = []
    for split in SPLITS:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        images = [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        labels = sorted(labels_dir.glob("*.txt"))
        if len(images) != EXPECTED_COUNTS[split] or len(labels) != EXPECTED_COUNTS[split]:
            raise RuntimeError(f"{split}: images={len(images)} labels={len(labels)} expected={EXPECTED_COUNTS[split]}")
        images_by_stem = {path.stem: path for path in images}
        labels_by_stem = {path.stem: path for path in labels}
        if set(images_by_stem) != set(labels_by_stem):
            missing = sorted(set(images_by_stem) - set(labels_by_stem))
            orphan = sorted(set(labels_by_stem) - set(images_by_stem))
            raise RuntimeError(f"{split}: missing_labels={missing[:10]} orphan_labels={orphan[:10]}")
        for label_path in labels:
            image_path = images_by_stem.get(label_path.stem)
            if image_path is None:
                raise FileNotFoundError(f"Label không có ảnh: {label_path}")
            issues, cleaned_text = repair_label(label_path)
            modified = modified_year is not None and (datetime.fromtimestamp(label_path.stat().st_mtime).year == modified_year or datetime.fromtimestamp(image_path.stat().st_mtime).year == modified_year)
            if issues or modified:
                reasons = (["label_issue"] if issues else []) + ([f"modified_{modified_year}"] if modified else [])
                records.append({"split": split, "label": str(label_path.relative_to(dataset_root)), "image": str(image_path.relative_to(dataset_root)), "issues": issues, "reasons": reasons, "cleaned_text": cleaned_text})
    return records


def split_validation(dataset_root):
    train_images = dataset_root / "train/images"
    train_labels = dataset_root / "train/labels"
    valid_images = dataset_root / "valid/images"
    valid_labels = dataset_root / "valid/labels"
    if valid_images.exists() or valid_labels.exists():
        if len(list(valid_images.glob("*"))) == EXPECTED_COUNTS["valid"] and len(list(valid_labels.glob("*.txt"))) == EXPECTED_COUNTS["valid"]:
            return 0
        raise RuntimeError("valid đã tồn tại nhưng số file không đúng")
    images_by_stem = {path.stem: path for path in train_images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    labels_by_stem = {path.stem: path for path in train_labels.glob("*.txt")}
    expected_total = EXPECTED_COUNTS["train"] + EXPECTED_COUNTS["valid"]
    if len(images_by_stem) != expected_total or len(labels_by_stem) != expected_total or set(images_by_stem) != set(labels_by_stem):
        raise RuntimeError(f"Không thể tách valid: images={len(images_by_stem)} labels={len(labels_by_stem)} expected={expected_total}")
    groups = {}
    for stem, label_path in labels_by_stem.items():
        class_ids = tuple(sorted({int(line.split()[0]) for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()}))
        groups.setdefault(class_ids, []).append(stem)
    target = EXPECTED_COUNTS["valid"]
    exact = {key: target * len(stems) / expected_total for key, stems in groups.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    for key in sorted(groups, key=lambda item: (exact[item] - quotas[item], str(item)), reverse=True)[:target - sum(quotas.values())]:
        quotas[key] += 1
    rng = random.Random(SPLIT_SEED)
    selected = []
    for key in sorted(groups, key=str):
        stems = sorted(groups[key])
        rng.shuffle(stems)
        selected.extend(stems[:quotas[key]])
    valid_images.mkdir(parents=True)
    valid_labels.mkdir(parents=True)
    for stem in sorted(selected):
        os.replace(images_by_stem[stem], valid_images / images_by_stem[stem].name)
        os.replace(labels_by_stem[stem], valid_labels / labels_by_stem[stem].name)
    return len(selected)


def audit_dataset(dataset_root):
    result = []
    for split in (name for name in SPLITS if (dataset_root / name).exists()):
        images = [path for path in (dataset_root / split / "images").iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        labels = sorted((dataset_root / split / "labels").glob("*.txt"))
        images_by_stem = {path.stem: path for path in images}
        labels_by_stem = {path.stem: path for path in labels}
        boxes = empty = clip = drop = affected = modified = 0
        for label_path in labels:
            source_text = label_path.read_text(encoding="utf-8")
            issues, _ = repair_label(label_path, source_text)
            boxes += sum(bool(line.strip()) for line in source_text.splitlines())
            empty += not bool(source_text.strip())
            clip += sum(item["action"] == "clip_edges" for item in issues)
            drop += sum(item["action"] == "drop_zero_area" for item in issues)
            affected += bool(issues)
            image_path = images_by_stem.get(label_path.stem)
            modified += image_path is not None and (datetime.fromtimestamp(label_path.stat().st_mtime).year == CURRENT_YEAR or datetime.fromtimestamp(image_path.stat().st_mtime).year == CURRENT_YEAR)
        result.append({"split": split, "images": len(images), "labels": len(labels), "boxes": boxes, "empty_labels": empty, "missing_labels": len(set(images_by_stem) - set(labels_by_stem)), "orphan_labels": len(set(labels_by_stem) - set(images_by_stem)), "affected_files": affected, "clip_boxes": clip, "drop_boxes": drop, f"modified_{CURRENT_YEAR}_samples": modified})
    return result


def apply_repairs(dataset_root, report_path):
    records = scan_dataset(dataset_root, CURRENT_YEAR)
    for record in records:
        if not record["issues"]:
            continue
        label_path = dataset_root / record["label"]
        temporary = label_path.with_suffix(".tmp")
        temporary.write_text(record["cleaned_text"], encoding="utf-8")
        os.replace(temporary, label_path)
    report = [{key: value for key, value in record.items() if key != "cleaned_text"} for record in records]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(".tmp")
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_report, report_path)
    remaining = scan_dataset(dataset_root)
    if remaining:
        raise RuntimeError(f"Còn {len(remaining)} file lỗi sau sửa")
    return report


def read_boxes(label_path):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), tuple(float(value) for value in parts[1:])))
    return boxes


def fit_display_size(width, height, available_width, available_height):
    scale = min(1.0, available_width / width, available_height / height)
    display_width = max(1, round(width * scale))
    display_height = max(1, round(height * scale))
    if display_width > width or display_height > height:
        raise AssertionError("Viewer không được upscale ảnh")
    if abs(display_width * height - display_height * width) > max(width, height):
        raise AssertionError("Viewer làm sai aspect ratio quá 1 pixel")
    return display_width, display_height


class ReviewApp:
    def __init__(self, dataset_root, records, report_path):
        self.dataset_root = dataset_root
        self.records = records
        self.report_path = report_path
        self.photo = None
        self.image = None
        self.image_size = None
        self.display_size = None
        self.origin = None
        self.boxes = []
        self.selected_box = None
        self.current_index = None
        self.drag = None
        self.history = []
        self.dirty = False
        self.saved_records = {index for index, record in enumerate(records) if record.get("reviewed")}
        self.root = tk.Tk()
        self.root.title("D-Fire label review")
        self.root.geometry("1100x720")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.mode = tk.StringVar(value="edit")
        self.class_id = tk.IntVar(value=0)
        self.listbox = tk.Listbox(self.root, width=39, exportselection=False)
        self.listbox.pack(side="left", fill="y", padx=8, pady=8)
        panel = tk.Frame(self.root)
        panel.pack(side="right", fill="both", expand=True, padx=(0, 8), pady=8)
        toolbar = tk.Frame(panel)
        toolbar.pack(fill="x", pady=(0, 6))
        tk.Radiobutton(toolbar, text="Chọn/sửa", variable=self.mode, value="edit").pack(side="left")
        tk.Radiobutton(toolbar, text="Thêm bbox", variable=self.mode, value="add").pack(side="left")
        tk.Label(toolbar, text="  Class:").pack(side="left")
        for class_id in sorted(CLASS_IDS):
            tk.Radiobutton(toolbar, text=f"{class_id} {CLASS_NAMES[class_id]}", variable=self.class_id, value=class_id, foreground=CLASS_COLORS[class_id]).pack(side="left")
        tk.Button(toolbar, text="Đổi nhãn", command=self.change_class).pack(side="left", padx=(6, 0))
        tk.Button(toolbar, text="Xóa", command=self.delete_selected).pack(side="left", padx=(4, 0))
        tk.Button(toolbar, text="Hoàn tác", command=self.undo).pack(side="left", padx=(4, 0))
        tk.Button(toolbar, text="Lưu", command=self.save).pack(side="right")
        self.canvas = tk.Canvas(panel, background="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(panel, anchor="w", justify="left", height=5, wraplength=700)
        self.status.pack(fill="x", pady=(6, 0))
        for index, record in enumerate(records, 1):
            self.listbox.insert("end", self.record_text(index - 1))
        self.listbox.bind("<<ListboxSelect>>", self.select_record)
        self.canvas.bind("<Configure>", self.render)
        self.canvas.bind("<ButtonPress-1>", self.mouse_press)
        self.canvas.bind("<B1-Motion>", self.mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_release)
        for key, delta in (("<Left>", -1), ("<Up>", -1), ("<Right>", 1), ("<Down>", 1)):
            self.root.bind(key, lambda event, step=delta: self.key_move(step))
            self.listbox.bind(key, lambda event, step=delta: self.key_move(step))
        self.root.bind("<Delete>", lambda event: self.delete_selected())
        self.root.bind("<Control-s>", lambda event: self.save())
        self.root.bind("<Control-z>", lambda event: self.undo())
        self.root.bind("a", lambda event: self.mode.set("add"))
        self.root.bind("e", lambda event: self.mode.set("edit"))
        self.root.bind("0", lambda event: self.class_id.set(0))
        self.root.bind("1", lambda event: self.class_id.set(1))
        self.listbox.selection_set(0)
        self.root.after(50, self.select_record)

    @staticmethod
    def issue_summary(record):
        clip = sum(item["action"] == "clip_edges" for item in record["issues"])
        drop = sum(item["action"] == "drop_zero_area" for item in record["issues"])
        parts = []
        if clip:
            parts.append(f"{clip} bbox vượt biên")
        if drop:
            parts.append(f"{drop} bbox không còn diện tích")
        if not parts:
            parts.append(f"mtime {CURRENT_YEAR}, không lỗi bbox")
        return ", ".join(parts)

    def record_text(self, index):
        record = self.records[index]
        saved = "✓ ĐÃ LƯU | " if index in self.saved_records else ""
        return f"{index + 1:03d} {saved}{record['split']}/{Path(record['label']).name} | {self.issue_summary(record)}"

    def refresh_record_text(self, index):
        self.listbox.delete(index)
        self.listbox.insert(index, self.record_text(index))
        self.listbox.selection_set(index)
        self.listbox.see(index)

    @staticmethod
    def issue_details(record):
        details = []
        for item in record["issues"]:
            class_name = CLASS_NAMES[item["class_id"]]
            if item["action"] == "clip_edges":
                details.append(f"Dòng {item['line']}, {class_name}: vượt biên ảnh → cắt mép vào [0,1].")
            else:
                details.append(f"Dòng {item['line']}, {class_name}: ngoài ảnh/không còn diện tích sau cắt → xóa.")
        return " ".join(details) or f"Không lỗi bbox; chỉ lọc do mtime ảnh/nhãn năm {CURRENT_YEAR}."

    @staticmethod
    def box_from_edges(x1, y1, x2, y2):
        left, right = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        top, bottom = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        return ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)

    @staticmethod
    def edges(box):
        x, y, width, height = box
        return (x - width / 2, y - height / 2, x + width / 2, y + height / 2)

    def remember(self):
        self.history.append([(class_id, tuple(box)) for class_id, box in self.boxes])
        self.history = self.history[-100:]

    def load_record(self, index):
        if self.dirty:
            self.save()
        self.current_index = index
        record = self.records[index]
        with Image.open(self.dataset_root / record["image"]) as source:
            self.image = source.convert("RGB")
        self.image_size = self.image.size
        self.boxes = read_boxes(self.dataset_root / record["label"])
        self.selected_box = None
        self.history = []
        self.dirty = False
        self.render()

    def select_record(self, event=None):
        if not self.listbox.curselection():
            return
        index = self.listbox.curselection()[0]
        if index != self.current_index:
            self.load_record(index)

    def move(self, delta):
        current = self.current_index if self.current_index is not None else 0
        target = max(0, min(len(self.records) - 1, current + delta))
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(target)
        self.listbox.see(target)
        self.load_record(target)

    def key_move(self, delta):
        self.move(delta)
        return "break"

    def normalized_point(self, event, clamp=False):
        if self.origin is None or self.display_size is None:
            return None
        x = (event.x - self.origin[0]) / self.display_size[0]
        y = (event.y - self.origin[1]) / self.display_size[1]
        if clamp:
            return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
        return (x, y) if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 else None

    def canvas_edges(self, box):
        x1, y1, x2, y2 = self.edges(box)
        return (
            self.origin[0] + x1 * self.display_size[0],
            self.origin[1] + y1 * self.display_size[1],
            self.origin[0] + x2 * self.display_size[0],
            self.origin[1] + y2 * self.display_size[1],
        )

    def hit_test(self, event):
        if self.selected_box is not None:
            x1, y1, x2, y2 = self.canvas_edges(self.boxes[self.selected_box][1])
            for name, x, y in (("nw", x1, y1), ("ne", x2, y1), ("sw", x1, y2), ("se", x2, y2)):
                if abs(event.x - x) <= 7 and abs(event.y - y) <= 7:
                    return self.selected_box, name
        point = self.normalized_point(event)
        if point is None:
            return None, None
        matches = []
        for index, (_, box) in enumerate(self.boxes):
            x1, y1, x2, y2 = self.edges(box)
            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                matches.append((box[2] * box[3], index))
        if not matches:
            return None, None
        return min(matches)[1], "select"

    def mouse_press(self, event):
        point = self.normalized_point(event)
        if point is None:
            self.selected_box = None
            self.render()
            return
        if self.mode.get() == "add":
            self.drag = {"kind": "add", "start": point, "current": point}
            self.render()
            return
        index, kind = self.hit_test(event)
        self.selected_box = index
        if index is not None:
            self.class_id.set(self.boxes[index][0])
            if kind != "select":
                self.drag = {"kind": kind, "start": point, "original": tuple(self.boxes[index][1]), "before": [(class_id, tuple(box)) for class_id, box in self.boxes]}
        self.render()

    def mouse_move(self, event):
        if self.drag is None:
            return
        point = self.normalized_point(event, clamp=True)
        if self.drag["kind"] == "add":
            self.drag["current"] = point
            self.render()
            return
        index = self.selected_box
        original = self.drag["original"]
        x1, y1, x2, y2 = self.edges(original)
        minimum_x = 2 / self.display_size[0]
        minimum_y = 2 / self.display_size[1]
        if "w" in self.drag["kind"]:
            x1 = min(point[0], x2 - minimum_x)
        else:
            x2 = max(point[0], x1 + minimum_x)
        if "n" in self.drag["kind"]:
            y1 = min(point[1], y2 - minimum_y)
        else:
            y2 = max(point[1], y1 + minimum_y)
        updated = self.box_from_edges(x1, y1, x2, y2)
        self.boxes[index] = (self.boxes[index][0], updated)
        self.render()

    def mouse_release(self, event):
        if self.drag is None:
            return
        if self.drag["kind"] == "add":
            start = self.drag["start"]
            end = self.normalized_point(event, clamp=True)
            box = self.box_from_edges(start[0], start[1], end[0], end[1])
            if box[2] * self.display_size[0] >= 2 and box[3] * self.display_size[1] >= 2:
                self.remember()
                self.boxes.append((self.class_id.get(), box))
                self.selected_box = len(self.boxes) - 1
                self.dirty = True
        else:
            before = self.drag["before"]
            if before != self.boxes:
                self.history.append(before)
                self.history = self.history[-100:]
                self.dirty = True
        self.drag = None
        self.render()

    def change_class(self):
        if self.selected_box is None:
            return
        class_id, box = self.boxes[self.selected_box]
        if class_id == self.class_id.get():
            return
        self.remember()
        self.boxes[self.selected_box] = (self.class_id.get(), box)
        self.dirty = True
        self.render()

    def delete_selected(self):
        if self.selected_box is None:
            return
        self.remember()
        del self.boxes[self.selected_box]
        self.selected_box = None
        self.dirty = True
        self.render()

    def undo(self):
        if not self.history:
            return
        self.boxes = self.history.pop()
        self.selected_box = None
        self.dirty = True
        self.render()

    def save(self):
        if self.current_index is None:
            return
        if self.dirty:
            label_path = self.dataset_root / self.records[self.current_index]["label"]
            lines = [f"{class_id} " + " ".join(f"{value:.12g}" for value in box) for class_id, box in self.boxes]
            temporary = label_path.with_suffix(".tmp")
            temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            os.replace(temporary, label_path)
        self.records[self.current_index]["reviewed"] = True
        temporary_report = self.report_path.with_suffix(".tmp")
        temporary_report.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_report, self.report_path)
        self.saved_records.add(self.current_index)
        self.dirty = False
        self.refresh_record_text(self.current_index)
        self.render()

    def close(self):
        self.save()
        self.root.destroy()

    def render(self, event=None):
        if self.image is None or self.current_index is None:
            return
        record = self.records[self.current_index]
        width, height = self.image_size
        canvas_width, canvas_height = max(self.canvas.winfo_width(), 200), max(self.canvas.winfo_height(), 200)
        display_width, display_height = fit_display_size(width, height, canvas_width - 24, canvas_height - 24)
        shown = self.image if (display_width, display_height) == self.image.size else self.image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(shown)
        origin_x = (canvas_width - display_width) / 2
        origin_y = (canvas_height - display_height) / 2
        self.display_size = (display_width, display_height)
        self.origin = (origin_x, origin_y)
        self.canvas.delete("all")
        self.canvas.create_image(origin_x, origin_y, anchor="nw", image=self.photo)
        for item in record["issues"]:
            x1, y1, x2, y2 = self.canvas_edges(item["original"])
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ff3030", width=2, dash=(6, 4))
            if item["fixed"] is None:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                self.canvas.create_line(cx - 6, cy - 6, cx + 6, cy + 6, fill="#ff3030", width=2)
                self.canvas.create_line(cx - 6, cy + 6, cx + 6, cy - 6, fill="#ff3030", width=2)
        for index, (class_id, box) in enumerate(self.boxes):
            x1, y1, x2, y2 = self.canvas_edges(box)
            color = CLASS_COLORS[class_id]
            selected = index == self.selected_box
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3 if selected else 2)
            self.canvas.create_text(x1 + 3, y1 + 3, anchor="nw", text=f"{class_id} {CLASS_NAMES[class_id]}", fill=color)
            if selected:
                for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                    self.canvas.create_rectangle(x - 4, y - 4, x + 4, y + 4, fill=color, outline="#ffffff")
        if self.drag is not None and self.drag["kind"] == "add":
            draft = self.box_from_edges(*self.drag["start"], *self.drag["current"])
            x1, y1, x2, y2 = self.canvas_edges(draft)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=CLASS_COLORS[self.class_id.get()], width=2, dash=(4, 3))
        fit_scale = min(1.0, (canvas_width - 24) / width, (canvas_height - 24) / height)
        selected = "không" if self.selected_box is None else f"#{self.selected_box + 1} {CLASS_NAMES[self.boxes[self.selected_box][0]]}"
        if self.dirty:
            saved = "CHƯA LƯU"
        elif self.current_index in self.saved_records:
            saved = "✓ ĐÃ LƯU"
        else:
            saved = "chưa đánh dấu lưu"
        self.status.configure(text=f"{self.current_index + 1}/{len(self.records)} | {record['label']} | gốc {width}×{height} | raster {display_width}×{display_height} | scale fit {fit_scale:.6f} | {saved}\n{self.issue_details(record)}\nCyan=smoke | cam=fire | đỏ nét đứt=bbox lỗi gốc | chọn={selected}\nA=thêm cả trong bbox khác | E=chọn/sửa | kéo góc=đổi kích thước | Delete=xóa | Ctrl+S=lưu thẳng dataset | Ctrl+Z=hoàn tác")

    def run(self):
        self.root.mainloop()


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", type=Path, default=repo_root / "datasets/smoke_fire_detection/D-Fire")
    parser.add_argument("--report", type=Path, default=repo_root / "artifacts/smoke_fire_detection/dfire_label_fixes.json")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--split-valid", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit_dataset(args.dataset), ensure_ascii=False, indent=2))
        return
    if args.split_valid:
        print({"moved_to_valid": split_validation(args.dataset), "seed": SPLIT_SEED})
    records = apply_repairs(args.dataset, args.report) if args.apply else json.loads(args.report.read_text(encoding="utf-8"))
    actions = {name: sum(item["action"] == name for record in records for item in record["issues"]) for name in ("clip_edges", "drop_zero_area")}
    print({"review_samples": len(records), "affected_files": sum(bool(record["issues"]) for record in records), f"modified_{CURRENT_YEAR}_samples": sum(f"modified_{CURRENT_YEAR}" in record["reasons"] for record in records), **actions})
    if not args.no_gui:
        ReviewApp(args.dataset, records, args.report).run()


if __name__ == "__main__":
    main()
