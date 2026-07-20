import json
import unittest
from pathlib import Path
from unittest.mock import patch

from tasks.smoke_fire_detection import eval as evalmod


class FakeDetection:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id


class ClassMapTests(unittest.TestCase):
    def test_single_class(self):
        class_map = evalmod.class_map_from_names(["smoke"])
        self.assertEqual(class_map["smoke_class_id"], 0)
        self.assertIsNone(class_map["fire_class_id"])
        self.assertEqual(class_map["names"], {"0": "smoke"})

    def test_two_class_order(self):
        class_map = evalmod.class_map_from_names(["smoke", "fire"])
        self.assertEqual(class_map["smoke_class_id"], 0)
        self.assertEqual(class_map["fire_class_id"], 1)

    def test_no_smoke_class_raises(self):
        with self.assertRaises(ValueError):
            evalmod.class_map_from_names(["fire"])


class DetectionSchemaTests(unittest.TestCase):
    def test_rfdetr_box_records_matches_yolo_schema_keys(self):
        detection = FakeDetection(
            xyxy=[[1.0, 2.0, 3.0, 4.0]],
            confidence=[0.5],
            class_id=[0],
        )
        records = evalmod.rfdetr_box_records(detection, ["smoke", "fire"])
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(set(record.keys()), {"class_id", "class_name", "confidence", "xyxy"})
        self.assertEqual(record["class_name"], "smoke")
        self.assertEqual(record["xyxy"], [1.0, 2.0, 3.0, 4.0])

    def test_rfdetr_box_records_empty_detection(self):
        detection = FakeDetection(xyxy=[], confidence=[], class_id=[])
        records = evalmod.rfdetr_box_records(detection, ["smoke"])
        self.assertEqual(records, [])

    def test_class_name_out_of_range_falls_back_to_id(self):
        detection = FakeDetection(xyxy=[[0, 0, 1, 1]], confidence=[0.9], class_id=[5])
        records = evalmod.rfdetr_box_records(detection, ["smoke"])
        self.assertEqual(records[0]["class_name"], "5")


class FrameKeyResumeTests(unittest.TestCase):
    def test_frame_key_is_hashable_and_stable(self):
        record = {"sequence_id": "seq", "timestamp_unix": "100", "ignition_offset_seconds": "-60"}
        key = evalmod.frame_key(record)
        self.assertEqual(key, ("seq", 100, -60))

    def test_load_resume_keys_rejects_duplicates(self, ):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write(json.dumps({"sequence_id": "a", "timestamp_unix": 1, "ignition_offset_seconds": 0}) + "\n")
            handle.write(json.dumps({"sequence_id": "a", "timestamp_unix": 1, "ignition_offset_seconds": 0}) + "\n")
            path = Path(handle.name)
        try:
            with self.assertRaises(ValueError):
                evalmod.load_resume_keys(path)
        finally:
            path.unlink()

    def test_resolve_frame_path_relative_uses_data_root(self):
        record = {"frame_path": "seq/frame.jpg"}
        resolved = evalmod.resolve_frame_path(record, Path("/data/root"))
        self.assertEqual(resolved, Path("/data/root/seq/frame.jpg"))

    def test_resolve_frame_path_absolute_ignores_data_root(self):
        record = {"frame_path": "/abs/seq/frame.jpg"}
        resolved = evalmod.resolve_frame_path(record, Path("/data/root"))
        self.assertEqual(resolved, Path("/abs/seq/frame.jpg"))


class ErrorMergeTests(unittest.TestCase):
    def test_load_existing_errors_missing_file(self):
        self.assertEqual(evalmod.load_existing_errors(Path("/nonexistent/path.errors.json")), [])

    def test_load_existing_errors_reads_json(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump([{"frame_path": "a.jpg", "error": "boom"}], handle)
            path = Path(handle.name)
        try:
            self.assertEqual(evalmod.load_existing_errors(path), [{"frame_path": "a.jpg", "error": "boom"}])
        finally:
            path.unlink()


class YoloLabelPathTests(unittest.TestCase):
    def test_replaces_last_images_segment(self):
        image_path = Path("/data/pyro-sdis-yolo/images/val/frame.jpg")
        label_path = evalmod.yolo_label_path(image_path)
        self.assertEqual(label_path, Path("/data/pyro-sdis-yolo/labels/val/frame.txt"))

    def test_nested_split_convention(self):
        image_path = Path("/data/D-Fire/test/images/frame.jpg")
        label_path = evalmod.yolo_label_path(image_path)
        self.assertEqual(label_path, Path("/data/D-Fire/test/labels/frame.txt"))

    def test_read_yolo_labels_converts_normalized_to_absolute_xywh(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("0 0.5 0.5 0.2 0.4\n")
            path = Path(handle.name)
        try:
            boxes = evalmod.read_yolo_labels(path, image_width=100, image_height=200)
            self.assertEqual(len(boxes), 1)
            self.assertEqual(boxes[0]["class_id"], 0)
            x_min, y_min, width, height = boxes[0]["bbox"]
            self.assertAlmostEqual(width, 20.0)
            self.assertAlmostEqual(height, 80.0)
            self.assertAlmostEqual(x_min, 40.0)
            self.assertAlmostEqual(y_min, 60.0)
        finally:
            path.unlink()

    def test_read_yolo_labels_missing_file_returns_empty(self):
        self.assertEqual(evalmod.read_yolo_labels(Path("/nonexistent.txt"), 100, 100), [])


class DetectorCacheBatchFallbackTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmpdir = Path(tempfile.mkdtemp())
        self.weights_path = self.tmpdir / "fake.pt"
        self.weights_path.write_bytes(b"fake")
        self.index_path = self.tmpdir / "index.jsonl"
        self.index_path.write_text("")
        self.data_root = self.tmpdir

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_args(self, out_path, resume=False):
        return type("Args", (), {
            "index": str(self.index_path),
            "weights": str(self.weights_path),
            "out": str(out_path),
            "backend": "yolo",
            "candidate_id": "fake",
            "class_names": "smoke",
            "conf": 0.05,
            "iou": 0.6,
            "imgsz": 640,
            "device": None,
            "limit": None,
            "data_root": str(self.data_root),
            "split": "all",
            "batch": 2,
            "resume": resume,
            "candidate_revision": "",
            "pilot": False,
            "sequence_ids": None,
        })()

    def _records(self):
        return [
            {"sequence_id": "s", "camera_id": "c", "frame_path": "a.jpg", "timestamp_unix": 1, "ignition_offset_seconds": 0, "weak_event_label": "negative", "split": "dev"},
            {"sequence_id": "s", "camera_id": "c", "frame_path": "b.jpg", "timestamp_unix": 2, "ignition_offset_seconds": 60, "weak_event_label": "negative", "split": "dev"},
        ]

    def test_batch_error_falls_back_to_per_frame(self):
        out_path = self.tmpdir / "cache.jsonl"
        args = self._make_args(out_path)

        def fake_predict_batch(paths):
            raise RuntimeError("batch failed")

        def fake_predict_one(path):
            return [{"class_id": 0, "class_name": "smoke", "confidence": 0.9, "xyxy": [0, 0, 1, 1]}]

        class_map = {"names": {"0": "smoke"}, "smoke_class_id": 0, "fire_class_id": None}
        fake_backend = (fake_predict_batch, fake_predict_one, class_map, "fake==1.0", "native", None)

        with patch.object(evalmod, "read_index", return_value=self._records()), \
             patch.object(evalmod, "load_backend", return_value=fake_backend):
            evalmod.cmd_detector_cache(args)

        lines = out_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            record = json.loads(line)
            self.assertEqual(record["max_smoke_confidence"], 0.9)

    def test_resume_writes_zero_new_frames(self):
        out_path = self.tmpdir / "cache.jsonl"
        args = self._make_args(out_path, resume=False)

        def fake_predict_batch(paths):
            return [[{"class_id": 0, "class_name": "smoke", "confidence": 0.4, "xyxy": [0, 0, 1, 1]}] for _ in paths]

        def fake_predict_one(path):
            return [{"class_id": 0, "class_name": "smoke", "confidence": 0.4, "xyxy": [0, 0, 1, 1]}]

        class_map = {"names": {"0": "smoke"}, "smoke_class_id": 0, "fire_class_id": None}
        fake_backend = (fake_predict_batch, fake_predict_one, class_map, "fake==1.0", "native", None)

        with patch.object(evalmod, "read_index", return_value=self._records()), \
             patch.object(evalmod, "load_backend", return_value=fake_backend):
            evalmod.cmd_detector_cache(args)
            first_run_lines = out_path.read_text(encoding="utf-8").splitlines()

            args_resume = self._make_args(out_path, resume=True)
            evalmod.cmd_detector_cache(args_resume)
            second_run_lines = out_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(first_run_lines), 2)
        self.assertEqual(len(second_run_lines), 2)


if __name__ == "__main__":
    unittest.main()
