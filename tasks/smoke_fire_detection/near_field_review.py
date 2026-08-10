import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


SOURCES = {
    "dfs_fire_v3": {
        "root": "datasets/smoke_fire_detection/dfs_fire_v3",
        "splits": ("train", "valid", "test"),
        "raw_to_canonical": {"0": "fire", "1": "smoke"},
    },
    "home_fire_2025": {
        "root": "datasets/smoke_fire_detection/home_fire_2025",
        "splits": ("train", "val", "test"),
        "raw_to_canonical": {"0": "fire", "1": "smoke"},
    },
    "indoor_fire_smoke_2025": {
        "root": "datasets/smoke_fire_detection/indoor_fire_smoke_2025/Indoor Fire Smoke",
        "splits": ("train", "valid", "test"),
        "raw_to_canonical": {"0": "fire", "1": "smoke"},
    },
    "annotated_fire_smoke_2025": {
        "root": "datasets/smoke_fire_detection/annotated_fire_smoke_2025",
        "splits": ("train", "valid", "test"),
        "raw_to_canonical": {"0": "fire", "1": "smoke"},
    },
    "fasdd_cv_v9": {
        "root": "datasets/smoke_fire_detection/fasdd_cv_v9/FASDD_CV",
        "splits": ("",),
        "raw_to_canonical": {"0": "fire", "1": "smoke"},
        "label_root": "annotations/YOLO_CV/labels",
    },
    "fire_and_smoke_v1": {
        "root": "datasets/smoke_fire_detection/fire_and_smoke_v1",
        "splits": ("train", "valid", "test"),
        "raw_to_canonical": {"0": "fire"},
        "force_review": True,
    },
    "deepquest_ai": {
        "root": "datasets/smoke_fire_detection/deepquest_ai/FIRE-SMOKE-DATASET",
        "kind": "image_classification",
        "splits": ("Train", "Test"),
        "raw_to_canonical": {"Fire": "fire", "Smoke": "smoke", "Neutral": "neutral"},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a stratified near-field label-review queue")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--queue")
    parser.add_argument("--decisions")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--source", action="append", choices=sorted(SOURCES))
    return parser.parse_args()



def summarize_decisions(args):
    queue = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    queued = {record["review_id"]: record for record in queue["records"]}
    allowed = {"pass", "missing", "wrong_or_extra", "uncertain"}
    seen = set()
    source_counts = defaultdict(lambda: defaultdict(int))
    issues = []
    for decision in decisions:
        review_id = decision.get("review_id")
        if review_id not in queued:
            raise ValueError(f"Unknown review_id: {review_id}")
        if review_id in seen:
            raise ValueError(f"Duplicate review_id: {review_id}")
        status = decision.get("status")
        if status not in allowed:
            raise ValueError(f"Invalid status for {review_id}: {status}")
        record = queued[review_id]
        if decision.get("image") != record["image"] or decision.get("source") != record["source"]:
            raise ValueError(f"Decision does not match queue: {review_id}")
        seen.add(review_id)
        source_counts[record["source"]][status] += 1
        if status != "pass":
            issues.append({
                "review_id": review_id,
                "source": record["source"],
                "source_split": record["source_split"],
                "category": record["category"],
                "status": status,
                "image": record["image"],
                "label": record["label"],
                "note": decision.get("note", ""),
            })
    missing = sorted(set(queued) - seen)
    if missing:
        raise ValueError(f"Missing decisions: {', '.join(missing[:5])}")
    output = {
        "purpose": "Validated round-1 human review outcome. It is evidence for data triage, not an automatic relabeling instruction.",
        "queue": args.queue,
        "decisions": args.decisions,
        "records_reviewed": len(decisions),
        "status_counts": {status: sum(counts[status] for counts in source_counts.values()) for status in sorted(allowed)},
        "source_status_counts": {source: dict(sorted(counts.items())) for source, counts in sorted(source_counts.items())},
        "issues": issues,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Validated {len(decisions)} decisions and saved {output_path}")
def label_path(image_path, source_root, config):
    if "label_root" in config:
        return source_root / config["label_root"] / f"{image_path.stem}.txt"
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def categories(raw_classes, raw_to_canonical):
    canonical = {raw_to_canonical[value] for value in raw_classes}
    if not canonical:
        return "empty", []
    if canonical == {"fire"}:
        return "fire_only", ["fire"]
    if canonical == {"smoke"}:
        return "smoke_only", ["smoke"]
    return "both", ["fire", "smoke"]


def scan_source(repo_root, source_name):
    config = SOURCES[source_name]
    source_root = repo_root / config["root"]
    records = []
    for split in config["splits"]:
        image_dir = source_root / split / "images" if split else source_root / "images"
        for image_path in sorted(image_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            raw_classes = []
            path = label_path(image_path, source_root, config)
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if fields:
                        raw_classes.append(fields[0])
            category, canonical_classes = categories(raw_classes, config["raw_to_canonical"])
            records.append({
                "source": source_name,
                "image": image_path.relative_to(repo_root).as_posix(),
                "label": path.relative_to(repo_root).as_posix(),
                "source_split": split or "unsplit",
                "category": category,
                "dataset_classes": canonical_classes,
                "review": {
                    "status": "pending",
                    "confirmed_classes": None,
                    "issue": None,
                    "note": None,
                },
            })
    return records


def main():
    args = parse_args()
    if args.decisions or args.queue:
        if not args.decisions or not args.queue:
            raise ValueError("--queue and --decisions must be used together")
        summarize_decisions(args)
        return
    repo_root = Path(args.repo_root).resolve()
    source_names = args.source or sorted(SOURCES)
    rng = random.Random(args.seed)
    selected = []
    coverage = {}
    for source_name in source_names:
        groups = defaultdict(list)
        for record in scan_source(repo_root, source_name):
            groups[record["category"]].append(record)
        coverage[source_name] = {}
        for category in ("fire_only", "smoke_only", "both", "empty"):
            choices = groups[category]
            rng.shuffle(choices)
            picked = choices[:args.per_category]
            coverage[source_name][category] = {"available": len(choices), "selected": len(picked)}
            selected.extend(picked)
    selected.sort(key=lambda item: (item["source"], item["category"], item["image"]))
    for index, record in enumerate(selected, start=1):
        record["review_id"] = f"nf-{index:04d}"
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "purpose": "Round-1 human label audit. This is a review queue, not a benchmark or training split.",
        "seed": args.seed,
        "per_category": args.per_category,
        "coverage": coverage,
        "records": selected,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(selected)} review records to {output}")


if __name__ == "__main__":
    main()
