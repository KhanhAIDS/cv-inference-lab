from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ModuleNotFoundError:
    sys.exit("Thiếu tkinter. Linux: cài gói python3-tk; Windows: dùng Python chính thức có Tcl/Tk.")

from PIL import Image, ImageTk


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LABEL_PATH_RE = re.compile(r"(datasets/.+?/labels/.+?\.txt)\s*$")
CLASS_NAMES = {0: "smoke", 1: "fire"}


@dataclass
class QueueItem:
    line: str
    label_path: Path
    image_path: Path


@dataclass
class Box:
    class_id: int
    x: float
    y: float
    w: float
    h: float


class LabelEditor:
    def __init__(self, root: tk.Tk, queue_path: Path) -> None:
        self.root = root
        self.queue_path = queue_path
        self.queue_lines = queue_path.read_text(encoding="utf-8").splitlines()
        self.items = self.parse_items()
        if not self.items:
            raise RuntimeError("Không có entry label hợp lệ trong file hàng đợi.")
        self.index = 0
        self.boxes: list[Box] = []
        self.selected: int | None = None
        self.dirty = False
        self.image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.image_x = 0.0
        self.image_y = 0.0
        self.display_w = 1.0
        self.display_h = 1.0
        self.drag_mode: str | None = None
        self.drag_origin: tuple[float, float] | None = None
        self.drag_box: Box | None = None
        self.redraw_after: str | None = None
        self.class_value = tk.StringVar(value="0")
        self.status_value = tk.StringVar()
        self.build_ui()
        self.load_current()

    def parse_items(self) -> list[QueueItem]:
        items: list[QueueItem] = []
        for line in self.queue_lines:
            match = LABEL_PATH_RE.search(line)
            if not match:
                continue
            label_path = (self.queue_path.parent / match.group(1)).resolve()
            image_path = self.find_image(label_path)
            if image_path is None:
                continue
            items.append(QueueItem(line=line, label_path=label_path, image_path=image_path))
        return items

    @staticmethod
    def find_image(label_path: Path) -> Path | None:
        image_dir = label_path.parent.parent / "images"
        for candidate in image_dir.glob(f"{label_path.stem}.*"):
            if candidate.suffix.lower() in IMAGE_SUFFIXES:
                return candidate
        return None

    def build_ui(self) -> None:
        self.root.title("D-Fire Label Editor")
        self.root.minsize(900, 650)
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Control-s>", lambda _: self.save_current())
        self.root.bind("<Delete>", lambda _: self.delete_selected())
        self.root.bind("<Left>", lambda _: self.navigate(-1))
        self.root.bind("<Right>", lambda _: self.navigate(1))

        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        ttk.Label(outer, textvariable=self.status_value).grid(row=0, column=0, sticky="w", pady=(0, 6))

        body = ttk.Frame(outer)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(body, background="#202020", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self.schedule_redraw)
        self.canvas.bind("<ButtonPress-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)

        side = ttk.Frame(body, padding=(8, 0, 0, 0))
        side.grid(row=0, column=1, sticky="ns")
        ttk.Label(side, text="Nhãn bbox").pack(anchor="w")
        self.class_box = ttk.Combobox(side, textvariable=self.class_value, values=("0", "1"), width=12)
        self.class_box.pack(fill=tk.X, pady=(2, 6))
        ttk.Button(side, text="Đổi nhãn bbox chọn", command=self.change_selected_class).pack(fill=tk.X)
        ttk.Button(side, text="Xóa bbox chọn", command=self.delete_selected).pack(fill=tk.X, pady=(6, 14))
        ttk.Label(side, text="BBox").pack(anchor="w")
        self.box_list = tk.Listbox(side, width=28, height=20, exportselection=False)
        self.box_list.pack(fill=tk.BOTH, expand=True)
        self.box_list.bind("<<ListboxSelect>>", self.select_from_list)
        ttk.Label(side, text="Kéo nền: vẽ\nKéo bbox: di chuyển\nKéo góc: đổi kích thước\nDel: xóa | Ctrl+S: lưu").pack(anchor="w", pady=(12, 0))

        controls = ttk.Frame(outer)
        controls.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(controls, text="← Trước", command=lambda: self.navigate(-1)).pack(side=tk.LEFT)
        ttk.Button(controls, text="Sau →", command=lambda: self.navigate(1)).pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Lưu", command=self.save_current).pack(side=tk.RIGHT)
        ttk.Button(controls, text="Chấp nhận", command=self.accept_current).pack(side=tk.RIGHT, padx=(0, 6))

    @property
    def current(self) -> QueueItem:
        return self.items[self.index]

    def load_current(self) -> None:
        self.image = Image.open(self.current.image_path).convert("RGB")
        self.boxes = self.read_boxes(self.current.label_path)
        self.selected = None
        self.dirty = False
        self.update_status()
        self.update_box_list()
        self.redraw()

    @staticmethod
    def read_boxes(path: Path) -> list[Box]:
        boxes: list[Box] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.split()
            if len(fields) != 5:
                continue
            try:
                class_id = int(fields[0])
                x, y, w, h = (float(value) for value in fields[1:])
            except ValueError:
                continue
            if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                boxes.append(Box(class_id, x, y, w, h))
            else:
                print(f"Bỏ qua label ngoài miền {path}:{line_number}", file=sys.stderr)
        return boxes

    def update_status(self) -> None:
        item = self.current
        self.status_value.set(f"{self.index + 1}/{len(self.items)}   {item.image_path.name}   {item.label_path}")

    def update_box_list(self) -> None:
        self.box_list.delete(0, tk.END)
        for index, box in enumerate(self.boxes):
            name = CLASS_NAMES.get(box.class_id, f"class {box.class_id}")
            self.box_list.insert(tk.END, f"{index + 1}: {box.class_id} ({name})  {box.x:.3f}, {box.y:.3f}, {box.w:.3f}, {box.h:.3f}")
        if self.selected is not None and self.selected < len(self.boxes):
            self.box_list.selection_set(self.selected)
            self.box_list.see(self.selected)

    def schedule_redraw(self, _: tk.Event[tk.Misc] | None = None) -> None:
        if self.redraw_after is not None:
            self.root.after_cancel(self.redraw_after)
        self.redraw_after = self.root.after(40, self.redraw)

    def redraw(self) -> None:
        self.redraw_after = None
        if self.image is None:
            return
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        image_w, image_h = self.image.size
        scale = min(canvas_w / image_w, canvas_h / image_h)
        self.display_w = max(image_w * scale, 1)
        self.display_h = max(image_h * scale, 1)
        self.image_x = (canvas_w - self.display_w) / 2
        self.image_y = (canvas_h - self.display_h) / 2
        resized = self.image.resize((round(self.display_w), round(self.display_h)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(self.image_x, self.image_y, image=self.photo, anchor=tk.NW)
        for index, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self.box_to_canvas(box)
            color = "#00e5ff" if box.class_id == 0 else "#ff7043"
            width = 4 if index == self.selected else 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)
            name = CLASS_NAMES.get(box.class_id, f"class {box.class_id}")
            self.canvas.create_text(x1 + 4, y1 + 4, text=f"{box.class_id}: {name}", fill="white", anchor=tk.NW, font=("TkDefaultFont", 10, "bold"))
            if index == self.selected:
                for x in (x1, x2):
                    for y in (y1, y2):
                        self.canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5, fill=color, outline="white")

    def box_to_canvas(self, box: Box) -> tuple[float, float, float, float]:
        x1 = self.image_x + (box.x - box.w / 2) * self.display_w
        y1 = self.image_y + (box.y - box.h / 2) * self.display_h
        x2 = self.image_x + (box.x + box.w / 2) * self.display_w
        y2 = self.image_y + (box.y + box.h / 2) * self.display_h
        return x1, y1, x2, y2

    def canvas_to_normalized(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.image_x) / self.display_w, (y - self.image_y) / self.display_h)

    def in_image(self, x: float, y: float) -> bool:
        return self.image_x <= x <= self.image_x + self.display_w and self.image_y <= y <= self.image_y + self.display_h

    def hit_test(self, x: float, y: float) -> tuple[int | None, str | None]:
        for index in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.box_to_canvas(self.boxes[index])
            corners = {"top_left": (x1, y1), "top_right": (x2, y1), "bottom_left": (x1, y2), "bottom_right": (x2, y2)}
            for corner, (corner_x, corner_y) in corners.items():
                if abs(x - corner_x) <= 9 and abs(y - corner_y) <= 9:
                    return index, corner
            if x1 <= x <= x2 and y1 <= y <= y2:
                return index, "move"
        return None, None

    def mouse_down(self, event: tk.Event[tk.Misc]) -> None:
        if not self.in_image(event.x, event.y):
            return
        index, mode = self.hit_test(event.x, event.y)
        self.selected = index
        self.drag_mode = mode or "draw"
        self.drag_origin = self.canvas_to_normalized(event.x, event.y)
        self.drag_box = None if index is None else Box(**vars(self.boxes[index]))
        self.update_box_list()
        self.redraw()

    def mouse_move(self, event: tk.Event[tk.Misc]) -> None:
        if self.drag_mode is None or self.drag_origin is None:
            return
        x, y = self.canvas_to_normalized(event.x, event.y)
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        origin_x, origin_y = self.drag_origin
        if self.drag_mode == "draw":
            class_id = self.parse_class_id()
            if class_id is None:
                return
            candidate = self.make_box(class_id, origin_x, origin_y, x, y)
            if self.selected is None:
                self.boxes.append(candidate)
                self.selected = len(self.boxes) - 1
            else:
                self.boxes[self.selected] = candidate
        elif self.selected is not None and self.drag_box is not None:
            if self.drag_mode == "move":
                dx, dy = x - origin_x, y - origin_y
                box = self.drag_box
                new_x = min(max(box.x + dx, box.w / 2), 1 - box.w / 2)
                new_y = min(max(box.y + dy, box.h / 2), 1 - box.h / 2)
                self.boxes[self.selected] = Box(box.class_id, new_x, new_y, box.w, box.h)
            else:
                x1, y1, x2, y2 = self.box_edges(self.drag_box)
                if "left" in self.drag_mode:
                    x1 = x
                else:
                    x2 = x
                if "top" in self.drag_mode:
                    y1 = y
                else:
                    y2 = y
                self.boxes[self.selected] = self.make_box(self.drag_box.class_id, x1, y1, x2, y2)
        self.dirty = True
        self.redraw()

    def mouse_up(self, _: tk.Event[tk.Misc]) -> None:
        if self.drag_mode == "draw" and self.selected is not None:
            box = self.boxes[self.selected]
            if box.w < 0.002 or box.h < 0.002:
                del self.boxes[self.selected]
                self.selected = None
                self.dirty = False
        self.drag_mode = None
        self.drag_origin = None
        self.drag_box = None
        self.update_box_list()
        self.redraw()

    @staticmethod
    def make_box(class_id: int, x1: float, y1: float, x2: float, y2: float) -> Box:
        left, right = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
        top, bottom = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
        return Box(class_id, (left + right) / 2, (top + bottom) / 2, right - left, bottom - top)

    @staticmethod
    def box_edges(box: Box) -> tuple[float, float, float, float]:
        return box.x - box.w / 2, box.y - box.h / 2, box.x + box.w / 2, box.y + box.h / 2

    def parse_class_id(self) -> int | None:
        try:
            class_id = int(self.class_value.get())
        except ValueError:
            messagebox.showerror("Nhãn không hợp lệ", "Nhãn phải là số nguyên, ví dụ 0 hoặc 1.")
            return None
        if class_id < 0:
            messagebox.showerror("Nhãn không hợp lệ", "Nhãn phải >= 0.")
            return None
        return class_id

    def select_from_list(self, _: tk.Event[tk.Misc]) -> None:
        selection = self.box_list.curselection()
        if not selection:
            return
        self.selected = selection[0]
        self.class_value.set(str(self.boxes[self.selected].class_id))
        self.redraw()

    def change_selected_class(self) -> None:
        if self.selected is None:
            messagebox.showinfo("Chưa chọn bbox", "Chọn một bbox trước.")
            return
        class_id = self.parse_class_id()
        if class_id is None:
            return
        box = self.boxes[self.selected]
        self.boxes[self.selected] = Box(class_id, box.x, box.y, box.w, box.h)
        self.dirty = True
        self.update_box_list()
        self.redraw()

    def delete_selected(self) -> None:
        if self.selected is None:
            return
        del self.boxes[self.selected]
        self.selected = None
        self.dirty = True
        self.update_box_list()
        self.redraw()

    def save_current(self) -> bool:
        if not self.dirty:
            return True
        content = "".join(f"{box.class_id} {box.x:.6f} {box.y:.6f} {box.w:.6f} {box.h:.6f}\n" for box in self.boxes)
        path = self.current.label_path
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
        self.dirty = False
        self.update_status()
        return True

    def resolve_dirty(self) -> bool:
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("Có thay đổi chưa lưu", "Có lưu label hiện tại trước khi chuyển ảnh?")
        if answer is None:
            return False
        if answer:
            return self.save_current()
        return True

    def navigate(self, offset: int) -> None:
        target = self.index + offset
        if not 0 <= target < len(self.items):
            return
        if not self.resolve_dirty():
            return
        self.index = target
        self.load_current()

    def accept_current(self) -> None:
        if not self.save_current():
            return
        accepted = self.current
        removed = False
        rewritten: list[str] = []
        for line in self.queue_lines:
            if not removed and line == accepted.line:
                removed = True
                continue
            rewritten.append(line)
        if not removed:
            messagebox.showerror("Không thể cập nhật hàng đợi", "Không tìm thấy entry hiện tại trong file hàng đợi.")
            return
        self.queue_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
        del self.items[self.index]
        self.queue_lines = rewritten
        if not self.items:
            messagebox.showinfo("Hoàn tất", "Không còn entry nào.")
            self.root.destroy()
            return
        self.index = min(self.index, len(self.items) - 1)
        self.load_current()

    def close(self) -> None:
        if self.resolve_dirty():
            self.root.destroy()


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    queue_path = root_dir / "d_fire_labels_modified_2026-07-07.txt"
    if not queue_path.is_file():
        sys.exit(f"Không tìm thấy: {queue_path}")
    root = tk.Tk()
    try:
        LabelEditor(root, queue_path)
    except RuntimeError as error:
        root.destroy()
        sys.exit(str(error))
    root.mainloop()


if __name__ == "__main__":
    main()
