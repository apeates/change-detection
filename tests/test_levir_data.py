import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from levir_data import DatasetError, audit_split, list_samples, load_pair


class LevirDataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.split = Path(directory.name)
        for folder in ("A", "B", "label"):
            (self.split / folder).mkdir()

    def write_pair(self, name, label):
        Image.new("RGB", (3, 2), "red").save(self.split / "A" / name)
        Image.new("RGB", (3, 2), "blue").save(self.split / "B" / name)
        Image.fromarray(np.array(label, dtype=np.uint8)).save(self.split / "label" / name)

    def test_threshold_preserves_light_gray_and_order(self):
        self.write_pair("pair.png", [[0, 127, 128], [156, 254, 255]])
        before, after, mask = load_pair(self.split, "pair.png")
        self.assertEqual(before.shape, (2, 3, 3))
        np.testing.assert_array_equal(before[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(after[0, 0], [0, 0, 255])
        self.assertEqual(mask.dtype, np.bool_)
        np.testing.assert_array_equal(mask, [[False, False, True], [True, True, True]])
        with Image.open(self.split / "label" / "pair.png") as label:
            self.assertEqual(label.getpixel((0, 1)), 156)

    def test_missing_pair_is_reported(self):
        self.write_pair("missing.png", [[0] * 3] * 2)
        (self.split / "B" / "missing.png").unlink()
        with self.assertRaisesRegex(DatasetError, r"missing\.png.*missing file in B"):
            list_samples(self.split)
        with self.assertRaisesRegex(DatasetError, r"Missing file: .*missing\.png"):
            load_pair(self.split, "missing.png")

    def test_mismatched_dimensions_are_reported(self):
        self.write_pair("mismatch.png", [[0] * 3] * 2)
        Image.new("RGB", (2, 2)).save(self.split / "B" / "mismatch.png")
        with self.assertRaisesRegex(DatasetError, r"mismatch\.png: dimension mismatch .*B=2x2"):
            load_pair(self.split, "mismatch.png")

    def test_unreadable_image_is_reported(self):
        self.write_pair("corrupt.png", [[0] * 3] * 2)
        (self.split / "A" / "corrupt.png").write_bytes(b"not a PNG")
        with self.assertRaisesRegex(DatasetError, r"Cannot read .*corrupt\.png"):
            load_pair(self.split, "corrupt.png")

    def test_audit_counts_unchanged_images(self):
        self.write_pair("z_empty.png", [[0] * 3] * 2)
        self.write_pair("a_changed.png", [[255, 0, 0], [0, 0, 0]])
        (self.split / "A" / ".DS_Store").touch()
        self.assertEqual(list_samples(self.split), ["a_changed.png", "z_empty.png"])
        self.assertEqual(audit_split(self.split), (2, 1))


if __name__ == "__main__":
    unittest.main()
