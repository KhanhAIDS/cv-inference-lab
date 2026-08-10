import argparse
import json
from collections import defaultdict
from pathlib import Path

from eval import load_backend
from near_field_review import SOURCES, categories, label_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a two-class detector and select label disagreements for review")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--backend", choices=["yolo", "rfdetr"], default="rfdetr")
    parser.add_argument("--weights")
    parser.add_argument("--output")
    parser.add_argument("--html", help="Offline reviewer HTML written beside the JSON queue")
    parser.add_argument("--queue", help="Existing model-audit queue to validate against decisions")
    parser.add_argument("--decisions", help="Downloaded reviewer decisions JSON")
    parser.add_argument("--source", action="append", choices=sorted(SOURCES))
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--missing-conf", type=float, default=0.70)
    parser.add_argument("--unsupported-conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--per-source", type=int, default=100)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def source_records(repo_root, source_name):
    config = SOURCES[source_name]
    source_root = repo_root / config["root"]
    records = []
    if config.get("kind") == "image_classification":
        for split in config["splits"]:
            for category_dir in sorted((source_root / split).iterdir()):
                if not category_dir.is_dir() or category_dir.name not in config["raw_to_canonical"]:
                    continue
                source_class = config["raw_to_canonical"][category_dir.name]
                for image_path in sorted(category_dir.glob("*")):
                    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue
                    records.append({
                        "source": source_name,
                        "source_split": split,
                        "image": image_path.relative_to(repo_root).as_posix(),
                        "label": None,
                        "label_classes": [] if source_class == "neutral" else [source_class],
                        "label_category": f"image_level_{source_class}",
                        "dataset_boxes": [],
                        "source_label_kind": "image_level",
                        "source_image_class": source_class,
                        "force_review": config.get("force_review", False),
                    })
        return records
    for split in config["splits"]:
        image_dir = source_root / split / "images" if split else source_root / "images"
        for image_path in sorted(image_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label = label_path(image_path, source_root, config)
            raw_classes = []
            dataset_boxes = []
            if label.exists():
                for line in label.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) == 5:
                        raw_classes.append(fields[0])
                        dataset_boxes.append({
                            "class": config["raw_to_canonical"][fields[0]],
                            "center_x": float(fields[1]),
                            "center_y": float(fields[2]),
                            "width": float(fields[3]),
                            "height": float(fields[4]),
                        })
            category, ground_truth = categories(raw_classes, config["raw_to_canonical"])
            records.append({
                "source": source_name,
                "source_split": split or "unsplit",
                "image": image_path.relative_to(repo_root).as_posix(),
                "label": label.relative_to(repo_root).as_posix(),
                "label_classes": ground_truth,
                "label_category": category,
                "dataset_boxes": dataset_boxes,
                "source_label_kind": "bbox",
                "force_review": config.get("force_review", False),
            })
    return records


def canonical_prediction(boxes, class_map):
    scores = {"smoke": 0.0, "fire": 0.0}
    canonical_boxes = []
    for box in boxes:
        class_id = int(box["class_id"])
        if class_id == class_map["smoke_class_id"]:
            label = "smoke"
        elif class_id == class_map["fire_class_id"]:
            label = "fire"
        else:
            continue
        score = float(box["confidence"])
        scores[label] = max(scores[label], score)
        canonical_boxes.append({"class": label, "confidence": score, "xyxy": box["xyxy"]})
    return scores, canonical_boxes


def disagreement(record, scores, missing_conf, unsupported_conf):
    expected = set(record["label_classes"])
    predicted = {name for name, score in scores.items() if score >= missing_conf}
    missing = sorted(predicted - expected)
    unsupported = sorted(name for name in expected if scores[name] <= unsupported_conf)
    if missing:
        return "possible_missing_label", missing
    if unsupported:
        return "possible_wrong_or_extra_label", unsupported
    return None, []


def write_review_html(records, output_path):
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html>
<meta charset="utf-8">
<title>Near-field model-assisted label review</title>
<style>
body { font: 16px system-ui, sans-serif; margin: 24px; max-width: 1200px; }
button, label { font: inherit; margin: 4px; padding: 10px; }
canvas { display: block; max-width: 100%; border: 1px solid #999; margin: 12px 0; }
textarea { width: 100%; height: 70px; }
#meta, #guide { color: #555; }
</style>
<h1>Model-assisted label review</h1>
<p id="guide">Đỏ liền = bbox nhãn fire, xanh liền = bbox nhãn smoke. Nét đứt = model, chỉ là gợi ý. “Duyệt bbox model” tạo nhãn dẫn xuất, không sửa dữ liệu gốc. Chỉ bấm khi bạn xác nhận bbox nét đứt đúng.</p>
<p id="meta"></p>
<canvas id="image"></canvas>
<div>
  <button onclick="markPass()">Nhãn đúng [1]</button>
  <label><input id="missing_or_extra" type="checkbox" onchange="toggleIssue('missing_or_extra')"> Thiếu/thừa bbox [2]</label>
  <label><input id="wrong_class" type="checkbox" onchange="toggleIssue('wrong_class')"> Sai class bbox [3]</label>
  <label><input id="geometry" type="checkbox" onchange="toggleIssue('geometry')"> Bbox sai vị trí/kích thước [4]</label>
  <button onclick="markUncertain()">Không chắc [5]</button>
  <button onclick="adoptTeacher()">Duyệt bbox model làm nhãn dẫn xuất [6]</button>
  <button onclick="excludeFromTraining()">Loại ảnh khỏi train [7]</button>
</div>
<textarea id="note" placeholder="Ghi chú ngắn nếu cần"></textarea>
<p><button onclick="previous()">Trước</button><button onclick="next()">Sau</button><button onclick="downloadDecisions()">Tải kết quả JSON</button></p>
<script>
const records = __PAYLOAD__;
let index = 0;
const storageKey = 'near_field_model_audit_review_state';
const state = JSON.parse(localStorage.getItem(storageKey) || '{}');
const canvas = document.querySelector('#image');
const context = canvas.getContext('2d');
const meta = document.querySelector('#meta');
const note = document.querySelector('#note');
const issueIds = ['missing_or_extra', 'wrong_class', 'geometry'];
function entry(record) { return state[record.review_id] || {status: 'pending', issues: [], label_action: 'keep_dataset', note: ''}; }
function save() { const record = records[index]; state[record.review_id] = {...entry(record), note: note.value}; localStorage.setItem(storageKey, JSON.stringify(state)); }
function render() {
  const record = records[index];
  const current = entry(record);
  note.value = current.note || '';
  for (const issue of issueIds) document.querySelector('#' + issue).checked = current.issues.includes(issue);
  const original = record.source_label_kind === 'image_level' ? `nhãn cấp ảnh: ${record.source_image_class}` : `bbox gốc: ${record.label_classes.join(', ') || 'empty'}`;
  meta.textContent = `${index + 1}/${records.length} · ${record.review_id} · ${record.source} · gợi ý: ${record.issue} (${record.classes_to_check.join(', ') || '—'}) · ${original}`;
  const image = new Image();
  image.onload = () => {
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    context.drawImage(image, 0, 0);
    const stroke = Math.max(2, image.naturalWidth / 450);
    for (const box of record.dataset_boxes) {
      context.setLineDash([]); context.strokeStyle = box.class === 'fire' ? '#ff3030' : '#00a8ff'; context.lineWidth = stroke;
      context.strokeRect((box.center_x - box.width / 2) * canvas.width, (box.center_y - box.height / 2) * canvas.height, box.width * canvas.width, box.height * canvas.height);
    }
    for (const box of record.teacher_boxes) {
      context.setLineDash([stroke * 3, stroke * 2]); context.strokeStyle = box.class === 'fire' ? '#9b0000' : '#005b8c'; context.lineWidth = stroke;
      context.strokeRect(box.xyxy[0], box.xyxy[1], box.xyxy[2] - box.xyxy[0], box.xyxy[3] - box.xyxy[1]);
      context.fillStyle = context.strokeStyle; context.font = `${Math.max(14, stroke * 5)}px system-ui`; context.fillText(`${box.class} ${box.confidence.toFixed(2)}`, box.xyxy[0], Math.max(16, box.xyxy[1] - 4));
    }
    context.setLineDash([]);
  };
  image.src = '../../' + record.image;
}
function markPass() { state[records[index].review_id] = {...entry(records[index]), status: 'pass', issues: [], label_action: 'keep_dataset'}; save(); next(); }
function markUncertain() { state[records[index].review_id] = {...entry(records[index]), status: 'uncertain', issues: [], label_action: 'needs_manual'}; save(); next(); }
function adoptTeacher() { const record = records[index]; if (!record.teacher_boxes.length) { alert('Model không có bbox để duyệt.'); return; } state[record.review_id] = {...entry(record), status: 'issue', issues: [...new Set([...entry(record).issues, 'missing_or_extra'])], label_action: 'adopt_teacher'}; save(); next(); }
function excludeFromTraining() { const record = records[index]; state[record.review_id] = {...entry(record), status: 'issue', issues: [...new Set([...entry(record).issues, 'missing_or_extra'])], label_action: 'exclude_from_training'}; save(); next(); }
function toggleIssue(issue) { const record = records[index]; const current = entry(record); const issues = new Set(current.issues); issues.has(issue) ? issues.delete(issue) : issues.add(issue); state[record.review_id] = {...current, status: issues.size ? 'issue' : 'pending', issues: [...issues]}; save(); }
function next() { save(); index = Math.min(index + 1, records.length - 1); render(); }
function previous() { save(); index = Math.max(index - 1, 0); render(); }
function downloadDecisions() { save(); const decisions = records.map(record => ({review_id: record.review_id, image: record.image, source: record.source, ...(entry(record))})); const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([JSON.stringify(decisions, null, 2)], {type: 'application/json'})); link.download = 'near_field_model_audit_decisions.json'; link.click(); }
addEventListener('keydown', event => { if (event.key === '1') markPass(); if (event.key === '2') toggleIssue('missing_or_extra'); if (event.key === '3') toggleIssue('wrong_class'); if (event.key === '4') toggleIssue('geometry'); if (event.key === '5') markUncertain(); if (event.key === 'ArrowRight') next(); if (event.key === 'ArrowLeft') previous(); });
render();
</script>
""".replace("__PAYLOAD__", payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def validate_decisions(queue_path, decisions_path, output_path):
    queue = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    decisions = json.loads(Path(decisions_path).read_text(encoding="utf-8"))
    queued = {record["review_id"]: record for record in queue["records"]}
    allowed_statuses = {"pass", "issue", "uncertain"}
    allowed_issues = {"missing_or_extra", "wrong_class", "geometry"}
    allowed_actions = {"keep_dataset", "adopt_teacher", "exclude_from_training", "needs_manual"}
    seen = set()
    source_counts = defaultdict(lambda: defaultdict(int))
    issue_counts = defaultdict(int)
    reviewed = []
    for decision in decisions:
        review_id = decision.get("review_id")
        if review_id not in queued or review_id in seen:
            raise ValueError(f"Invalid or duplicate review_id: {review_id}")
        record = queued[review_id]
        if decision.get("image") != record["image"] or decision.get("source") != record["source"]:
            raise ValueError(f"Decision does not match queue: {review_id}")
        status = decision.get("status")
        issues = decision.get("issues", [])
        action = decision.get("label_action", "keep_dataset")
        if status not in allowed_statuses or not isinstance(issues, list) or set(issues) - allowed_issues or action not in allowed_actions:
            raise ValueError(f"Invalid decision: {review_id}")
        if (status == "issue") != bool(issues):
            raise ValueError(f"Issue/status mismatch: {review_id}")
        if action == "adopt_teacher" and not record["teacher_boxes"]:
            raise ValueError(f"Cannot adopt empty teacher prediction: {review_id}")
        seen.add(review_id)
        source_counts[record["source"]][status] += 1
        for issue in issues:
            issue_counts[issue] += 1
        if status != "pass" or action != "keep_dataset":
            reviewed.append({
                "review_id": review_id,
                "source": record["source"],
                "source_split": record["source_split"],
                "image": record["image"],
                "label": record["label"],
                "status": status,
                "issues": issues,
                "label_action": action,
                "teacher_boxes": record["teacher_boxes"] if action == "adopt_teacher" else [],
                "note": decision.get("note", ""),
            })
    missing = sorted(set(queued) - seen)
    if missing:
        raise ValueError(f"Missing decisions: {', '.join(missing[:5])}")
    output = {
        "purpose": "Validated model-assisted human review outcome. Approved teacher boxes are labels for a future derived dataset only; raw datasets remain unchanged.",
        "queue": str(queue_path),
        "decisions": str(decisions_path),
        "records_reviewed": len(decisions),
        "status_counts": {status: sum(counts[status] for counts in source_counts.values()) for status in sorted(allowed_statuses)},
        "issue_counts": dict(sorted(issue_counts.items())),
        "label_action_counts": {action: sum(1 for decision in decisions if decision.get("label_action", "keep_dataset") == action) for action in sorted(allowed_actions)},
        "source_status_counts": {source: dict(sorted(counts.items())) for source, counts in sorted(source_counts.items())},
        "issues": reviewed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Validated {len(decisions)} decisions and saved {output_path}")


def main():
    args = parse_args()
    if args.queue or args.decisions:
        if not args.queue or not args.decisions or not args.output:
            raise ValueError("--queue, --decisions and --output must be used together")
        validate_decisions(args.queue, args.decisions, Path(args.output).resolve())
        return
    if not args.weights or not args.output:
        raise ValueError("--weights and --output are required to create a queue")
    repo_root = Path(args.repo_root).resolve()
    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(weights)
    if args.missing_conf <= args.unsupported_conf:
        raise ValueError("--missing-conf must be greater than --unsupported-conf")
    source_names = args.source or sorted(SOURCES)
    records = []
    for source_name in source_names:
        records.extend(source_records(repo_root, source_name))
    if args.limit:
        records = records[:args.limit]
    backend_args = argparse.Namespace(
        backend=args.backend,
        class_names="smoke,fire",
        conf=args.conf,
        iou=0.6,
        imgsz=args.imgsz,
        device=args.device,
    )
    predict_batch, _, class_map, framework, postprocess, _ = load_backend(backend_args, weights)
    if class_map["fire_class_id"] is None:
        raise ValueError("teacher must detect both smoke and fire")
    candidates = []
    for start in range(0, len(records), args.batch):
        batch = records[start:start + args.batch]
        boxes_per_image = predict_batch([repo_root / record["image"] for record in batch])
        for record, boxes in zip(batch, boxes_per_image):
            scores, canonical_boxes = canonical_prediction(boxes, class_map)
            kind, classes = disagreement(record, scores, args.missing_conf, args.unsupported_conf)
            if record["source_label_kind"] == "image_level" and record["source_image_class"] in {"fire", "smoke"} and scores[record["source_image_class"]] >= args.missing_conf:
                kind = "possible_pseudo_label"
                classes = [record["source_image_class"]]
            if record["force_review"]:
                kind = "full_source_semantic_review"
                classes = record["label_classes"] or ["fire"]
            if kind:
                candidates.append({
                    **record,
                    "issue": kind,
                    "classes_to_check": classes,
                    "teacher_scores": scores,
                    "teacher_boxes": canonical_boxes,
                })
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["source"]].append(candidate)
    selected = []
    for source_name in source_names:
        source_candidates = sorted(
            grouped[source_name],
            key=lambda item: max(item["teacher_scores"][name] for name in item["classes_to_check"]),
            reverse=True,
        )
        selected.extend(source_candidates[:args.per_source])
    for index, candidate in enumerate(selected, start=1):
        candidate["review_id"] = f"model-audit-{index:05d}"
    output = {
        "purpose": "Model-assisted review queue. Teacher predictions become derived labels only after explicit human approval; raw datasets are never edited.",
        "teacher": {
            "backend": args.backend,
            "weights": str(weights.relative_to(repo_root)),
            "framework": framework,
            "postprocess": postprocess,
            "class_names": class_map["names"],
            "conf": args.conf,
            "missing_conf": args.missing_conf,
            "unsupported_conf": args.unsupported_conf,
            "imgsz": args.imgsz,
        },
        "source_counts": {source: len(grouped[source]) for source in source_names},
        "records_selected": len(selected),
        "records": selected,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.html:
        write_review_html(selected, Path(args.html).resolve())
    print(f"Saved {len(selected)} model-audit records to {output_path}")


if __name__ == "__main__":
    main()
