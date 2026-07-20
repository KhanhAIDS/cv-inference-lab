import json
import random
import tempfile
import unittest
from pathlib import Path

from tasks.smoke_fire_detection import temporal_eval as tmod


def make_synthetic_cache(path, quality, seed, noise_scale=0.3):
    """Write a cache with `quality` in [0,1] controlling smoke/no-smoke separation.

    Signal-plus-noise model: separation between positive/negative score centers grows
    with quality against a fixed noise scale, so AUROC varies smoothly and
    monotonically with quality instead of saturating at 1.0 for any quality > 0.
    Same sequence/camera/timestamp/offset layout is reused across candidates so
    caches stay frame-key aligned; only max_smoke_confidence differs per candidate.
    """
    rng = random.Random(seed)
    n_sequences = 40
    lines = []
    for seq_index in range(n_sequences):
        sequence_id = f"seq{seq_index:03d}"
        camera_id = f"cam{seq_index % 10:02d}"
        for offset in (-300, -120, 200, 400):
            label = 1 if offset >= 180 else 0
            center = 0.5 + 0.4 * quality if label == 1 else 0.5 - 0.4 * quality
            score = max(0.0, min(1.0, rng.gauss(center, noise_scale)))
            lines.append(json.dumps({
                "sequence_id": sequence_id,
                "camera_id": camera_id,
                "timestamp_unix": 1000000 + offset,
                "ignition_offset_seconds": offset,
                "max_smoke_confidence": score,
            }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class AurocTieHandlingTests(unittest.TestCase):
    def test_all_scores_tied_gives_chance_auroc(self):
        # Every score identical -> ranks all tie to the mean rank -> AUROC must be exactly 0.5.
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [0, 1, 0, 1]
        self.assertEqual(tmod.auroc(scores, labels), 0.5)

    def test_partial_ties_match_mid_rank_correction(self):
        # Two negatives at 0.1, one positive at 0.1 (tied with negatives), one positive at 0.9.
        # Tied group at 0.1 spans ranks 1-3 (mean rank 2); the untied top score gets rank 4.
        scores = [0.1, 0.1, 0.1, 0.9]
        labels = [0, 0, 1, 1]
        n_pos, n_neg = 2, 2
        rank_sum_pos = 2 + 4  # tied positive gets mean rank 2, untied positive gets rank 4
        expected = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        self.assertAlmostEqual(tmod.auroc(scores, labels), expected)

    def test_single_class_returns_none(self):
        self.assertIsNone(tmod.auroc([0.1, 0.2, 0.3], [1, 1, 1]))
        self.assertIsNone(tmod.auroc([0.1, 0.2, 0.3], [0, 0, 0]))


class ComboBootstrapTests(unittest.TestCase):
    def test_combo_auroc_bootstrap_pairwise_matches_sign(self):
        groups = {
            "g1": [{"_label": 1, "a": 0.9, "b": 0.2}, {"_label": 0, "a": 0.1, "b": 0.8}],
            "g2": [{"_label": 1, "a": 0.85, "b": 0.3}, {"_label": 0, "a": 0.2, "b": 0.7}],
        }
        report = tmod.combo_auroc_bootstrap(groups, ["g1", "g2"], [(1, "a"), (-1, "b")], samples=200, seed=1)
        self.assertGreater(report["median"], 0)
        self.assertGreater(report["ci95"][0], -1)

    def test_combo_auroc_bootstrap_empty_when_single_class(self):
        groups = {"g1": [{"_label": 1, "a": 0.9}, {"_label": 1, "a": 0.8}]}
        report = tmod.combo_auroc_bootstrap(groups, ["g1"], [(1, "a")], samples=50, seed=1)
        self.assertEqual(report["n_resamples_used"], 0)
        self.assertIsNone(report["median"])


class BuildJoinedRecordsTests(unittest.TestCase):
    def test_common_keys_and_alignment_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            cache_a = tmpdir / "a.jsonl"
            cache_b = tmpdir / "b.jsonl"
            cache_a.write_text(
                json.dumps({"sequence_id": "s", "camera_id": "c", "timestamp_unix": 1, "ignition_offset_seconds": -300, "max_smoke_confidence": 0.1}) + "\n" +
                json.dumps({"sequence_id": "s", "camera_id": "c", "timestamp_unix": 2, "ignition_offset_seconds": 300, "max_smoke_confidence": 0.9}) + "\n",
                encoding="utf-8",
            )
            cache_b.write_text(
                json.dumps({"sequence_id": "s", "camera_id": "c", "timestamp_unix": 1, "ignition_offset_seconds": -300, "max_smoke_confidence": 0.2}) + "\n",
                encoding="utf-8",
            )
            joined, alignment = tmod.build_joined_records({"a": cache_a, "b": cache_b}, ignore_band_seconds=180)
            self.assertEqual(alignment["a"]["total_frames"], 2)
            self.assertEqual(alignment["a"]["missing_from_common"], 1)
            self.assertEqual(alignment["b"]["missing_from_common"], 0)
            self.assertEqual(len(joined), 1)


class CompareMatrixEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.caches = {}
        for candidate_id, quality, seed in [
            ("rfdetr_pyro", 0.95, 10),
            ("yolo_pyro", 0.55, 11),
            ("rfdetr_dfire", 0.45, 12),
            ("yolo_dfire", 0.20, 13),
        ]:
            path = self.tmpdir / f"{candidate_id}.jsonl"
            make_synthetic_cache(path, quality, seed)
            self.caches[candidate_id] = path

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_args(self, out_path):
        return type("Args", (), {
            "candidate": [f"{cid}={path}" for cid, path in self.caches.items()],
            "clean": ",".join(self.caches.keys()),
            "architecture": ["rfdetr_pyro=rfdetr", "yolo_pyro=yolo", "rfdetr_dfire=rfdetr", "yolo_dfire=yolo"],
            "dataset": ["rfdetr_pyro=pyro", "yolo_pyro=pyro", "rfdetr_dfire=dfire", "yolo_dfire=dfire"],
            "dataset_order": "pyro,dfire",
            "latency_ms": [],
            "out": str(out_path),
            "ignore_band_seconds": 180,
            "bootstrap_samples": 300,
            "seed": 20260707,
        })()

    def test_leader_is_highest_quality_candidate_and_wins(self):
        out_path = self.tmpdir / "matrix.json"
        tmod.cmd_compare_matrix(self._make_args(out_path))
        result = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(result["leader"], "rfdetr_pyro")
        self.assertEqual(result["winner"], "rfdetr_pyro")
        self.assertTrue(all(result["leader_beats_rival"].values()))
        self.assertEqual(len(result["pairwise"]), 6)

    def test_architecture_and_dataset_effects_have_expected_sign(self):
        out_path = self.tmpdir / "matrix.json"
        tmod.cmd_compare_matrix(self._make_args(out_path))
        result = json.loads(out_path.read_text(encoding="utf-8"))
        # rfdetr consistently built with higher quality than yolo at the same dataset
        self.assertGreater(result["architecture_effect_rfdetr_minus_yolo_by_dataset"]["pyro"]["point_value"], 0)
        self.assertGreater(result["architecture_effect_rfdetr_minus_yolo_by_dataset"]["dfire"]["point_value"], 0)
        # pyro built with higher quality than dfire at the same architecture
        self.assertGreater(result["dataset_effect_by_architecture"]["rfdetr"]["point_value"], 0)
        self.assertGreater(result["dataset_effect_by_architecture"]["yolo"]["point_value"], 0)
        self.assertIsNotNone(result["interaction"]["point_value"])

    def test_tie_breaks_by_latency_when_top_set_has_multiple(self):
        # Make all four candidates near-identical in quality so no leader statistically beats every rival.
        for candidate_id, seed in [("rfdetr_pyro", 20), ("yolo_pyro", 21), ("rfdetr_dfire", 22), ("yolo_dfire", 23)]:
            make_synthetic_cache(self.caches[candidate_id], 0.5, seed=seed)
        latency = {"rfdetr_pyro": 50.0, "yolo_pyro": 10.0, "rfdetr_dfire": 40.0, "yolo_dfire": 8.0}
        args = self._make_args(self.tmpdir / "matrix.json")
        args.latency_ms = [f"{cid}={ms}" for cid, ms in latency.items()]
        tmod.cmd_compare_matrix(args)
        result = json.loads(Path(args.out).read_text(encoding="utf-8"))
        if len(result["top_set"]) > 1:
            expected_winner = min(result["top_set"], key=lambda cid: latency[cid])
            self.assertEqual(result["winner"], expected_winner)
            self.assertIn("tie-break", result["winner_rule"])
        else:
            self.assertIsNotNone(result["winner"])


if __name__ == "__main__":
    unittest.main()
