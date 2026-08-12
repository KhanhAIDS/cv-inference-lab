import argparse
import gzip
import json
import mimetypes
import os
import shutil
import socket
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DATASET_MARKER = "datasets/smoke_fire_detection/"
CLASS_NAMES = ["smoke", "fire"]
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
TEACHER_BOX_CAP = 25
DECISION_STATUSES = ["draft", "confirmed", "excluded"]
LABEL_SOURCES = ["dataset", "teacher", "human"]
QUEUE_CLOSING_STATUSES = {"confirmed", "excluded"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GMT7 = timezone(timedelta(hours=7))
DEFAULT_WORKSPACE = "artifacts/smoke_fire_detection/label_audit"
DEFAULT_EXPORT = "datasets/smoke_fire_detection/near_field_curated_ground_truth"
TOOL_FILES = ["label_audit_tool.py", "label_audit_tool.html"]


def parse_args():
    parser = argparse.ArgumentParser(description="Merge model-audit bundles and run a local bbox annotation tool")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_workspace(command, help_text):
        node = sub.add_parser(command, help=help_text)
        node.add_argument("--workspace", default=DEFAULT_WORKSPACE)
        node.add_argument("--repo-root", default=".")
        return node

    importer = with_workspace("import", "Merge review bundles into one annotation workspace")
    importer.add_argument("--bundle", action="append", required=True, help="Bundle ZIP or extracted bundle directory")

    server = with_workspace("serve", "Serve the annotation tool over HTTP")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8760)

    exporter = with_workspace("export", "Write audited labels as a derived ground-truth tree")
    exporter.add_argument("--output", default=DEFAULT_EXPORT)
    exporter.add_argument("--link-images", action="store_true", help="Create <stem>.audited<ext> symlinks beside the labels")

    with_workspace("status", "Print audit progress")

    packer = with_workspace("pack", "Build a self-contained offline workspace with images included")
    packer.add_argument("--output", required=True, help="Target directory for the offline pack")
    packer.add_argument("--source", action="append")
    packer.add_argument("--issue", action="append")
    packer.add_argument("--limit", type=int)
    packer.add_argument("--max-per-source", type=int, default=0)
    packer.add_argument("--include-audited", action="store_true")
    packer.add_argument("--zip", action="store_true", help="Also write <output>.zip")

    merger = with_workspace("merge", "Merge decisions.jsonl produced on another machine")
    merger.add_argument("--decisions", required=True, help="decisions.jsonl from an offline pack")
    return parser.parse_args()


def now_gmt7():
    return datetime.now(GMT7).isoformat(timespec="seconds")


def resolve_path(repo_root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def dataset_relative(image_path):
    normalized = str(image_path).replace("\\", "/")
    index = normalized.find(DATASET_MARKER)
    if index < 0:
        raise ValueError(f"Image path does not contain {DATASET_MARKER}: {image_path}")
    return normalized[index:]


def clamp01(value):
    return min(1.0, max(0.0, float(value)))


def read_bundle(bundle):
    path = Path(bundle)
    if path.is_dir():
        candidates = sorted(path.rglob("queue.json"))
        if not candidates:
            raise FileNotFoundError(f"No queue.json inside {path}")
        return path.name, json.loads(candidates[0].read_text(encoding="utf-8"))
    with zipfile.ZipFile(path) as archive:
        candidates = sorted(name for name in archive.namelist() if name.endswith("queue.json"))
        if not candidates:
            raise FileNotFoundError(f"No queue.json inside {path}")
        return path.stem, json.loads(archive.read(candidates[0]).decode("utf-8"))


def bundle_records(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise ValueError("Unsupported queue payload")


def normalize_dataset_boxes(record):
    return [
        {
            "cls": box["class"],
            "cx": round(float(box["center_x"]), 6),
            "cy": round(float(box["center_y"]), 6),
            "w": round(float(box["width"]), 6),
            "h": round(float(box["height"]), 6),
        }
        for box in record.get("dataset_boxes", [])
        if box["class"] in CLASS_IDS
    ]


def normalize_teacher_boxes(record):
    width = float(record.get("image_width") or 0)
    height = float(record.get("image_height") or 0)
    if width <= 0 or height <= 0:
        return []
    boxes = []
    for box in record.get("teacher_boxes", []):
        if box["class"] not in CLASS_IDS:
            continue
        x1, y1, x2, y2 = (float(value) for value in box["xyxy"])
        boxes.append({
            "cls": box["class"],
            "conf": round(float(box["confidence"]), 4),
            "cx": round(clamp01((x1 + x2) / 2 / width), 6),
            "cy": round(clamp01((y1 + y2) / 2 / height), 6),
            "w": round(clamp01(abs(x2 - x1) / width), 6),
            "h": round(clamp01(abs(y2 - y1) / height), 6),
        })
    boxes.sort(key=lambda box: -box["conf"])
    return boxes[:TEACHER_BOX_CAP]


def normalize_record(record, bundle_name):
    image_rel = dataset_relative(record["image"])
    return {
        "task_id": image_rel[len(DATASET_MARKER):],
        "bundle": bundle_name,
        "queue_type": record.get("queue_type", "error"),
        "source": record["source"],
        "split": record.get("source_split", "unsplit"),
        "image": image_rel,
        "label": dataset_relative(record["label"]) if record.get("label") else None,
        "width": int(record.get("image_width") or 0),
        "height": int(record.get("image_height") or 0),
        "label_kind": record.get("source_label_kind", "bbox"),
        "image_class": record.get("source_image_class"),
        "issue": record.get("issue", "unspecified"),
        "secondary_issues": list(record.get("secondary_issues") or []),
        "static_issues": sorted({item["code"] for item in record.get("static_issues") or []}),
        "priority": round(float(record.get("priority") or 0.0), 4),
        "teacher_scores": {name: round(float(score), 4) for name, score in (record.get("teacher_scores") or {}).items()},
        "teacher_box_count": int(record.get("teacher_box_count") or 0),
        "dataset_boxes": normalize_dataset_boxes(record),
        "teacher_boxes": normalize_teacher_boxes(record),
        "bundle_image": None,
    }


def tasks_path(workspace):
    return workspace / "tasks.jsonl.gz"


def decisions_path(workspace):
    return workspace / "decisions.jsonl"


def load_tasks(workspace):
    path = tasks_path(workspace)
    if not path.exists():
        raise FileNotFoundError(f"Workspace has no queue: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_tasks(workspace, tasks):
    with gzip.open(tasks_path(workspace), "wt", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")


def read_decision_lines(path):
    decisions = {}
    if not Path(path).exists():
        return decisions
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                decision = json.loads(line)
                decisions[decision["task_id"]] = decision
    return decisions


def load_decisions(workspace):
    return read_decision_lines(decisions_path(workspace))


def append_decision(workspace, decision):
    with decisions_path(workspace).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(decision, ensure_ascii=False) + "\n")


def decision_status(decisions, task_id):
    decision = decisions.get(task_id)
    return decision.get("status", "pending") if decision else "pending"


def select_tasks(tasks, decisions, sources, issues, statuses, limit, max_per_source, order="priority"):
    selected = []
    for task in tasks:
        if sources and task["source"] not in sources:
            continue
        if issues and task["issue"] not in issues:
            continue
        if "all" not in statuses and decision_status(decisions, task["task_id"]) not in statuses:
            continue
        selected.append(task)
    if order == "priority":
        selected.sort(key=lambda task: (-task["priority"], task["task_id"]))
    elif order == "boxes":
        selected.sort(key=lambda task: (len(task["dataset_boxes"]), task["task_id"]))
    else:
        selected.sort(key=lambda task: task.get("order", 0))
    if max_per_source > 0:
        per_source = Counter()
        capped = []
        for task in selected:
            if per_source[task["source"]] >= max_per_source:
                continue
            per_source[task["source"]] += 1
            capped.append(task)
        selected = capped
    return selected[:limit] if limit else selected


def command_import(args):
    repo_root = Path(args.repo_root).resolve()
    workspace = resolve_path(repo_root, args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    previous = load_decisions(workspace)
    merged = {}
    duplicates = 0
    bundles = []
    missing_images = []
    for bundle in args.bundle:
        bundle_name, payload = read_bundle(bundle)
        records = bundle_records(payload)
        kept = 0
        for record in records:
            task = normalize_record(record, bundle_name)
            if not (repo_root / task["image"]).exists():
                missing_images.append(task["image"])
                continue
            if task["task_id"] in merged:
                duplicates += 1
                existing = merged[task["task_id"]]
                existing["secondary_issues"] = sorted(set(existing["secondary_issues"]) | {task["issue"]} | set(task["secondary_issues"]))
                existing["priority"] = max(existing["priority"], task["priority"])
                continue
            merged[task["task_id"]] = task
            kept += 1
        bundles.append({"bundle": bundle_name, "source_path": str(bundle), "records": len(records), "kept": kept})
    ordered = sorted(merged.values(), key=lambda task: (task["source"], -task["priority"], task["task_id"]))
    for index, task in enumerate(ordered, start=1):
        task["order"] = index
        task["carried_over_status"] = previous.get(task["task_id"], {}).get("status")
    write_tasks(workspace, ordered)
    meta = {
        "purpose": "Annotation workspace. decisions.jsonl is the authoritative human ground truth; raw datasets are never edited.",
        "created_at_gmt7": now_gmt7(),
        "classes": CLASS_NAMES,
        "teacher_box_cap": TEACHER_BOX_CAP,
        "bundles": bundles,
        "tasks": len(ordered),
        "duplicate_task_ids_merged": duplicates,
        "missing_images_skipped": len(missing_images),
        "already_audited_carried_over": sum(1 for task in ordered if task["carried_over_status"] in QUEUE_CLOSING_STATUSES),
        "source_counts": dict(sorted(Counter(task["source"] for task in ordered).items())),
        "issue_counts": dict(sorted(Counter(task["issue"] for task in ordered).items())),
    }
    (workspace / "workspace.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Merged {len(ordered)} tasks from {len(bundles)} bundles into {workspace}")
    print(f"  duplicates merged: {duplicates}; missing images: {len(missing_images)}; already audited: {meta['already_audited_carried_over']}")


def command_status(args):
    repo_root = Path(args.repo_root).resolve()
    workspace = resolve_path(repo_root, args.workspace)
    tasks = load_tasks(workspace)
    decisions = load_decisions(workspace)
    columns = ["pending"] + DECISION_STATUSES
    for title, key in (("source", "source"), ("issue", "issue")):
        grouped = defaultdict(Counter)
        for task in tasks:
            grouped[task[key]][decision_status(decisions, task["task_id"])] += 1
        print(f"{title:34s}" + "".join(f"{name:>11s}" for name in columns) + f"{'total':>8s}")
        for name in sorted(grouped):
            counts = grouped[name]
            print(f"{name:34s}" + "".join(f"{counts[column]:>11d}" for column in columns) + f"{sum(counts.values()):>8d}")
        print()
    total = Counter(decision_status(decisions, task["task_id"]) for task in tasks)
    print("total:", dict(sorted(total.items())), "of", len(tasks))
    confirmed = Counter(decision.get("label_source", "human") for decision in decisions.values() if decision.get("status") == "confirmed")
    print("confirmed label_source:", dict(sorted(confirmed.items())))


def exported_label_path(task_id):
    parts = task_id.split("/")
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    parts[-1] = f"{Path(parts[-1]).stem}.audited.txt"
    return "/".join(parts)


def command_export(args):
    repo_root = Path(args.repo_root).resolve()
    workspace = resolve_path(repo_root, args.workspace)
    output = resolve_path(repo_root, args.output)
    tasks = {task["task_id"]: task for task in load_tasks(workspace)}
    decisions = load_decisions(workspace)
    written = []
    class_counts = Counter()
    source_counts = Counter()
    label_source_counts = Counter()
    excluded = []
    link_failures = 0
    for task_id, decision in sorted(decisions.items()):
        status = decision.get("status")
        if status == "excluded":
            excluded.append(task_id)
            continue
        if status != "confirmed" or task_id not in tasks:
            continue
        task = tasks[task_id]
        boxes = [box for box in decision.get("boxes", []) if box["cls"] in CLASS_IDS]
        for box in boxes:
            class_counts[box["cls"]] += 1
        lines = [f"{CLASS_IDS[box['cls']]} {box['cx']:.6f} {box['cy']:.6f} {box['w']:.6f} {box['h']:.6f}" for box in boxes]
        label_file = output / exported_label_path(task_id)
        label_file.parent.mkdir(parents=True, exist_ok=True)
        label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        image_link = None
        if args.link_images:
            image_link = label_file.with_name(f"{Path(task_id).stem}.audited{Path(task_id).suffix}")
            try:
                if image_link.exists() or image_link.is_symlink():
                    image_link.unlink()
                os.symlink(repo_root / task["image"], image_link)
            except OSError:
                link_failures += 1
                image_link = None
        label_source = decision.get("label_source", "human")
        label_source_counts[label_source] += 1
        source_counts[task["source"]] += 1
        written.append({
            "task_id": task_id,
            "source": task["source"],
            "split": task["split"],
            "raw_image": task["image"],
            "raw_label": task["label"],
            "audited_label": exported_label_path(task_id),
            "audited_image_link": image_link.relative_to(output).as_posix() if image_link else None,
            "label_source": label_source,
            "boxes": len(lines),
            "eval_safe": label_source != "teacher",
            "decided_at_gmt7": decision.get("decided_at_gmt7"),
        })
    manifest = {
        "purpose": "Derived ground-truth labels from human audit. Raw datasets unchanged; every file name carries the .audited marker.",
        "eval_policy": "eval splits must use eval_safe records only; label_source=teacher is train-only because the teacher is rfdetr_large_dfire itself",
        "exported_at_gmt7": now_gmt7(),
        "class_ids": CLASS_IDS,
        "labels_written": len(written),
        "images_excluded": excluded,
        "box_class_counts": dict(sorted(class_counts.items())),
        "label_source_counts": dict(sorted(label_source_counts.items())),
        "eval_safe_count": sum(1 for record in written if record["eval_safe"]),
        "source_counts": dict(sorted(source_counts.items())),
        "symlink_failures": link_failures,
        "records": written,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "ground_truth_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported {len(written)} audited labels and {len(excluded)} exclusions to {output}")
    print(f"  label_source: {dict(sorted(label_source_counts.items()))}; eval_safe: {manifest['eval_safe_count']}")
    if link_failures:
        print(f"  symlink failures: {link_failures}")


PACK_README = """Label audit - offline pack

Windows:  bấm đúp START_WINDOWS.bat
Linux/mac: ./START_LINUX.sh

Cần Python 3.9+ (Windows dùng `py`). Không cần cài thêm package.
Sau đó mở http://127.0.0.1:8760/ nếu trình duyệt chưa tự mở.

Mọi quyết định ghi vào decisions.jsonl trong thư mục này.
Khi xong, copy decisions.jsonl về server và chạy:
  python tasks/smoke_fire_detection/label_audit_tool.py merge --decisions <duong-dan>/decisions.jsonl
"""

PACK_BAT = """@echo off
cd /d "%~dp0"
start "" http://127.0.0.1:8760/
py label_audit_tool.py serve --workspace . --repo-root . --host 127.0.0.1 --port 8760
pause
"""

PACK_SH = """#!/bin/sh
cd "$(dirname "$0")"
python3 label_audit_tool.py serve --workspace . --repo-root . --host 127.0.0.1 --port 8760
"""


def command_pack(args):
    repo_root = Path(args.repo_root).resolve()
    workspace = resolve_path(repo_root, args.workspace)
    output = resolve_path(repo_root, args.output)
    tasks = load_tasks(workspace)
    decisions = load_decisions(workspace)
    statuses = {"all"} if args.include_audited else {"pending", "draft"}
    selected = select_tasks(tasks, decisions, set(args.source or []), set(args.issue or []), statuses, args.limit, args.max_per_source)
    if not selected:
        raise SystemExit("No task matches the filters")
    images_root = output / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    packed = []
    for task in selected:
        source_file = repo_root / task["image"]
        if not source_file.exists():
            continue
        destination = images_root / task["task_id"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source_file, destination)
        total_bytes += destination.stat().st_size
        packed.append({**task, "bundle_image": f"images/{task['task_id']}", "carried_over_status": decision_status(decisions, task["task_id"])})
    write_tasks(output, packed)
    carried = {task["task_id"]: decisions[task["task_id"]] for task in packed if task["task_id"] in decisions}
    if carried:
        with decisions_path(output).open("w", encoding="utf-8") as handle:
            for decision in carried.values():
                handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    tool_dir = Path(__file__).resolve().parent
    for name in TOOL_FILES:
        shutil.copy2(tool_dir / name, output / name)
    (output / "README.txt").write_text(PACK_README, encoding="utf-8")
    (output / "START_WINDOWS.bat").write_text(PACK_BAT.replace("\n", "\r\n"), encoding="utf-8")
    launcher = output / "START_LINUX.sh"
    launcher.write_text(PACK_SH, encoding="utf-8")
    launcher.chmod(0o755)
    meta = {
        "purpose": "Self-contained offline annotation pack. Images included; no server or dataset access needed.",
        "created_at_gmt7": now_gmt7(),
        "classes": CLASS_NAMES,
        "tasks": len(packed),
        "image_bytes": total_bytes,
        "filters": {"source": args.source, "issue": args.issue, "limit": args.limit, "max_per_source": args.max_per_source, "include_audited": bool(args.include_audited)},
        "source_counts": dict(sorted(Counter(task["source"] for task in packed).items())),
        "issue_counts": dict(sorted(Counter(task["issue"] for task in packed).items())),
        "return_instructions": "Copy decisions.jsonl back and run: label_audit_tool.py merge --decisions <path>",
    }
    (output / "workspace.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Packed {len(packed)} tasks ({total_bytes / 1e6:.0f} MB of images) into {output}")
    if args.zip:
        archive = shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)
        print(f"  archive: {archive} ({Path(archive).stat().st_size / 1e6:.0f} MB)")


def command_merge(args):
    repo_root = Path(args.repo_root).resolve()
    workspace = resolve_path(repo_root, args.workspace)
    known = {task["task_id"] for task in load_tasks(workspace)}
    current = load_decisions(workspace)
    incoming = read_decision_lines(resolve_path(repo_root, args.decisions))
    unknown = sorted(task_id for task_id in incoming if task_id not in known)
    if unknown:
        raise SystemExit(f"{len(unknown)} unknown task_id, first: {unknown[0]}")
    added = 0
    updated = 0
    for task_id, decision in incoming.items():
        if decision.get("status") not in DECISION_STATUSES:
            raise SystemExit(f"Invalid status for {task_id}")
        existing = current.get(task_id)
        if existing == decision:
            continue
        append_decision(workspace, decision)
        if existing is None:
            added += 1
        else:
            updated += 1
    merged = load_decisions(workspace)
    print(f"Merged {len(incoming)} decisions into {workspace}: {added} new, {updated} updated, {len(incoming) - added - updated} unchanged")
    print("total by status:", dict(sorted(Counter(decision["status"] for decision in merged.values()).items())))


class AuditHandler(BaseHTTPRequestHandler):
    repo_root = None
    workspace = None
    tasks = []
    task_index = {}
    decisions = {}
    page = b""
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *fmt_args):
        return

    def send_payload(self, body, content_type, cache=False):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload, code=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def progress(self):
        per_source = defaultdict(Counter)
        for task in self.tasks:
            per_source[task["source"]][decision_status(self.decisions, task["task_id"])] += 1
        return {
            "overall": dict(sorted(Counter(decision_status(self.decisions, task["task_id"]) for task in self.tasks).items())),
            "per_source": {source: dict(sorted(counts.items())) for source, counts in sorted(per_source.items())},
        }

    def resolve_image(self, task):
        for candidate in (self.workspace / task["bundle_image"] if task.get("bundle_image") else None, self.repo_root / task["image"]):
            if candidate is None:
                continue
            resolved = candidate.resolve()
            if resolved.exists() and (str(resolved).startswith(str(self.workspace)) or str(resolved).startswith(str(self.repo_root))):
                return resolved
        return None

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path in {"/", "/index.html"}:
            self.send_payload(self.page, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/bootstrap":
            self.send_json({
                "classes": CLASS_NAMES,
                "statuses": DECISION_STATUSES,
                "total_tasks": len(self.tasks),
                "sources": dict(sorted(Counter(task["source"] for task in self.tasks).items())),
                "issues": dict(sorted(Counter(task["issue"] for task in self.tasks).items())),
                "progress": self.progress()["overall"],
            })
            return
        if parsed.path == "/api/tasks":
            statuses = set(query.get("status", [])) or {"pending", "draft"}
            selected = select_tasks(
                self.tasks, self.decisions,
                set(query.get("source", [])), set(query.get("issue", [])), statuses,
                None, int(query.get("max_per_source", ["0"])[0]),
                query.get("order", ["priority"])[0],
            )
            limit = int(query.get("limit", ["500"])[0])
            payload = [
                {**task, "status": decision_status(self.decisions, task["task_id"]), "decision": self.decisions.get(task["task_id"])}
                for task in selected[:limit]
            ]
            self.send_json({"tasks": payload, "matched": len(selected), "returned": len(payload)})
            return
        if parsed.path == "/api/progress":
            self.send_json(self.progress())
            return
        if parsed.path == "/image":
            task = self.task_index.get(query.get("task", [""])[0])
            path = self.resolve_image(task) if task else None
            if path is None:
                self.send_json({"error": "image not found"}, code=404)
                return
            self.send_payload(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream", cache=True)
            return
        self.send_json({"error": "not found"}, code=404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/decision":
            self.send_json({"error": "not found"}, code=404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_json({"error": "invalid json"}, code=400)
            return
        task_id = body.get("task_id")
        if task_id not in self.task_index:
            self.send_json({"error": "unknown task_id"}, code=400)
            return
        if body.get("status") not in DECISION_STATUSES:
            self.send_json({"error": "invalid status"}, code=400)
            return
        label_source = body.get("label_source", "human")
        if label_source not in LABEL_SOURCES:
            self.send_json({"error": "invalid label_source"}, code=400)
            return
        boxes = []
        for box in body.get("boxes", []):
            if box.get("cls") not in CLASS_IDS:
                self.send_json({"error": "invalid class"}, code=400)
                return
            cx, cy, width, height = (clamp01(box.get(key, 0.0)) for key in ("cx", "cy", "w", "h"))
            if width <= 0 or height <= 0:
                continue
            boxes.append({"cls": box["cls"], "cx": round(cx, 6), "cy": round(cy, 6), "w": round(width, 6), "h": round(height, 6)})
        task = self.task_index[task_id]
        decision = {
            "task_id": task_id,
            "source": task["source"],
            "split": task["split"],
            "image": task["image"],
            "status": body["status"],
            "label_source": label_source,
            "boxes": boxes,
            "seconds_spent": round(float(body.get("seconds_spent") or 0.0), 1),
            "queue_issue": task["issue"],
            "decided_at_gmt7": now_gmt7(),
        }
        append_decision(self.workspace, decision)
        self.decisions[task_id] = decision
        self.send_json({"saved": True, "status": decision["status"], "boxes": len(boxes), "label_source": label_source})


def command_serve(args):
    repo_root = Path(args.repo_root).resolve()
    workspace = resolve_path(repo_root, args.workspace)
    page = Path(__file__).with_name("label_audit_tool.html")
    if not page.exists():
        raise FileNotFoundError(page)
    tasks = load_tasks(workspace)
    AuditHandler.repo_root = repo_root
    AuditHandler.workspace = workspace
    AuditHandler.tasks = tasks
    AuditHandler.task_index = {task["task_id"]: task for task in tasks}
    AuditHandler.decisions = load_decisions(workspace)
    AuditHandler.page = page.read_bytes()
    server = ThreadingHTTPServer((args.host, args.port), AuditHandler)
    print(f"Loaded {len(tasks)} tasks and {len(AuditHandler.decisions)} decisions from {workspace}")
    print(f"  http://127.0.0.1:{args.port}/")
    if args.host not in {"127.0.0.1", "localhost"}:
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            print(f"  http://{probe.getsockname()[0]}:{args.port}/")
            probe.close()
        except OSError:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        server.server_close()


def main():
    args = parse_args()
    {"import": command_import, "serve": command_serve, "export": command_export,
     "status": command_status, "pack": command_pack, "merge": command_merge}[args.command](args)


if __name__ == "__main__":
    main()
