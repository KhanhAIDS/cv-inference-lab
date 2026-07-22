import argparse
import json
from pathlib import Path

from temporal_eval import auroc, auroc_bootstrap_ci


def read_cache(cache_path: Path):
    records = []
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def gate_decision(auc):
    if auc is None:
        return "undecided"
    if auc >= 0.80:
        return "pass"
    if auc >= 0.60:
        return "marginal"
    return "fail"


def category_report(records, category, score_key, bootstrap_samples, seed):
    subset = [r for r in records if r["category"] == category]
    scores = [r[score_key] for r in subset]
    labels = [r["label"] for r in subset]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    auc = auroc(scores, labels)
    groups = {r["video_id"]: [{"_label": r["label"], score_key: r[score_key]}] for r in subset}
    bootstrap = auroc_bootstrap_ci(groups, sorted(groups), score_key, bootstrap_samples, seed) if auc is not None else None
    return {
        "category": category,
        "score_key": score_key,
        "n_videos": len(subset),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auroc": auc,
        "bootstrap_video": bootstrap,
        "decision": gate_decision(auc),
    }


def negative_confidence_summary(records, category, score_key):
    negatives = [r for r in records if r["category"] == category and r["label"] == 0]
    scores = sorted(r[score_key] for r in negatives)
    n = len(scores)
    if n == 0:
        return {"category": category, "n_negative_videos": 0}
    thresholds = [0.05, 0.25, 0.5]
    return {
        "category": category,
        "n_negative_videos": n,
        "mean": sum(scores) / n,
        "median": scores[n // 2],
        "max": scores[-1],
        "min": scores[0],
        "n_above_threshold": {str(t): sum(1 for s in scores if s >= t) for t in thresholds},
        "videos_above_0.5": [r["video_id"] for r in negatives if r[score_key] >= 0.5],
    }


def main():
    parser = argparse.ArgumentParser(description="AUROC gate + trick-negative report for FIRESENSE video-level detector cache (near-field control, not FIgLib schema)")
    parser.add_argument("--cache", required=True, help="Path to firesense_detector_cache_*.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()

    cache_path = Path(args.cache).resolve()
    records = read_cache(cache_path)

    smoke_report = category_report(records, "smoke", "max_smoke_confidence", args.bootstrap_samples, args.seed)
    fire_report = category_report(records, "fire", "max_fire_confidence", args.bootstrap_samples, args.seed + 1)

    pooled_scores = []
    pooled_labels = []
    pooled_groups = {}
    for r in records:
        score_key = "max_smoke_confidence" if r["category"] == "smoke" else "max_fire_confidence"
        pooled_scores.append(r[score_key])
        pooled_labels.append(r["label"])
        pooled_groups[r["video_id"]] = [{"_label": r["label"], "_score": r[score_key]}]
    pooled_auc = auroc(pooled_scores, pooled_labels)
    pooled_bootstrap = auroc_bootstrap_ci(pooled_groups, sorted(pooled_groups), "_score", args.bootstrap_samples, args.seed + 2) if pooled_auc is not None else None

    trick_negative = {
        "definition": "toan bo negative FIRESENSE (25/49 video) la trick-negative theo thiet ke dataset goc (Zenodo 836749, ghi trong research_plan.md muc 10.5) -- khong co tap negative 'thuong' rieng biet de doi chieu",
        "smoke_videos_neg": negative_confidence_summary(records, "smoke", "max_smoke_confidence"),
        "fire_videos_neg": negative_confidence_summary(records, "fire", "max_fire_confidence"),
    }

    result = {
        "cache": str(cache_path),
        "n_videos_total": len(records),
        "n_pos": sum(r["label"] for r in records),
        "n_neg": sum(1 - r["label"] for r in records),
        "pooled_auroc_primary_class": pooled_auc,
        "pooled_bootstrap_video": pooled_bootstrap,
        "pooled_decision": gate_decision(pooled_auc),
        "by_category": {
            "smoke": smoke_report,
            "fire": fire_report,
        },
        "trick_negative_report_NG2": trick_negative,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"FIRESENSE gate report saved: {out_path}")
    print(f"pooled AUROC (primary class per video) = {pooled_auc}")


if __name__ == "__main__":
    main()
