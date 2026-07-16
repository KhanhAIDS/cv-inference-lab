import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from tasks.smoke_fire_detection.dataset import cmd_pyro_sdis


class PyroSdisConverterTest(unittest.TestCase):
    def image_bytes(self):
        image = Image.new("RGB", (1280, 720), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def write_shard(self, root, name, records):
        image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
        table = pa.table({
            "image": pa.array([{"bytes": record["image"], "path": None} for record in records], type=image_type),
            "annotations": [record["annotations"] for record in records],
            "image_name": [record["image_name"] for record in records],
        })
        pq.write_table(table, root / name)

    def convert(self, root):
        cmd_pyro_sdis(argparse.Namespace(
            data_root=str(root),
            out=str(root / "out"),
            audit_out=str(root / "audit.json"),
        ))

    def test_valid_multiple_boxes_and_empty_annotation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = self.image_bytes()
            self.write_shard(root, "train-00000-of-00001.parquet", [
                {"image": image, "annotations": "1 0.5 0.5 0.2 0.4\n1 0.2 0.2 0.1 0.1", "image_name": "a.jpg"},
                {"image": image, "annotations": "", "image_name": "b.jpg"},
            ])
            self.write_shard(root, "val-00000-of-00001.parquet", [
                {"image": image, "annotations": "1 0.5 0.5 0.2 0.4", "image_name": "c.jpg"},
            ])
            self.convert(root)
            audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["images"], {"train": 2, "val": 1})
            self.assertEqual(audit["empty_annotations"], 1)
            self.assertEqual(audit["bbox_count"], 3)
            self.assertEqual((root / "out" / "labels" / "train" / "a.txt").read_text(encoding="utf-8").split()[0], "0")

    def test_invalid_inputs_leave_no_output(self):
        cases = [
            {"image": self.image_bytes(), "annotations": "0 0.5 0.5 0.2 0.4", "image_name": "bad-class.jpg"},
            {"image": self.image_bytes(), "annotations": "1 1.2 0.5 0.2 0.4", "image_name": "bad-box.jpg"},
            {"image": b"corrupt", "annotations": "1 0.5 0.5 0.2 0.4", "image_name": "bad-image.jpg"},
        ]
        for record in cases:
            with self.subTest(record=record["image_name"]), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.write_shard(root, "train-00000-of-00001.parquet", [record])
                with self.assertRaises(ValueError):
                    self.convert(root)
                self.assertFalse((root / "out").exists())

    def test_duplicate_filename_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = {"image": self.image_bytes(), "annotations": "1 0.5 0.5 0.2 0.4", "image_name": "same.jpg"}
            self.write_shard(root, "train-00000-of-00001.parquet", [record])
            self.write_shard(root, "val-00000-of-00001.parquet", [record])
            with self.assertRaises(ValueError):
                self.convert(root)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
