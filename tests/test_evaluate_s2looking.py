import importlib.util
import io
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
from PIL import Image


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install requirements-train.txt to test U-Net")
class S2LookingTestArchiveTests(unittest.TestCase):
    def write_png(self, archive, path, pixels):
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        archive.writestr(path, buffer.getvalue())

    def test_uses_test_split_and_unions_directional_labels(self):
        from evaluate_s2looking import S2LookingTest

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        added = np.zeros((8, 8), dtype=np.uint8)
        removed = added.copy()
        added[1, 2] = 1
        removed[3, 4] = 255
        with zipfile.ZipFile(io.BytesIO(), "w") as archive, \
             patch("evaluate_s2looking.TEST_PAIR_COUNT", 1), \
             patch("evaluate_s2looking.IMAGE_SHAPE", (8, 8)):
            self.write_png(archive, "S2Looking/train/Image1/train.png", rgb)
            for folder, pixels in (("Image1", rgb), ("Image2", rgb), ("label1", added), ("label2", removed)):
                self.write_png(archive, f"S2Looking/test/{folder}/example.png", pixels)
            dataset = S2LookingTest(archive)
            self.assertEqual(dataset.samples, ["example.png"])
            _, _, reference = dataset.load_pair("example.png")
            np.testing.assert_array_equal(reference, (added != 0) | (removed != 0))

            combined = np.maximum(added, removed)
            self.write_png(archive, "S2Looking/test/label/example.png", combined)
            dataset = S2LookingTest(archive)
            self.assertEqual(dataset.labels, ("label",))
            np.testing.assert_array_equal(dataset.load_pair("example.png")[2], reference)

    def test_rejects_mismatched_names(self):
        from evaluate_s2looking import S2LookingTest
        from levir_data import DatasetError

        with zipfile.ZipFile(io.BytesIO(), "w") as archive, patch("evaluate_s2looking.TEST_PAIR_COUNT", 1):
            archive.writestr("test/Image1/first.png", b"")
            archive.writestr("test/Image2/second.png", b"")
            archive.writestr("test/label/first.png", b"")
            with self.assertRaisesRegex(DatasetError, "matched S2Looking test pairs"):
                S2LookingTest(archive)

    def test_unions_colored_directional_labels_and_rejects_other_colors(self):
        from evaluate_s2looking import S2LookingSplit
        from levir_data import DatasetError

        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        added, removed = rgb.copy(), rgb.copy()
        added[1, 2] = (255, 0, 0)
        removed[3, 4] = (0, 0, 255)
        expected = np.zeros((8, 8), dtype=bool)
        expected[1, 2] = expected[3, 4] = True
        with zipfile.ZipFile(io.BytesIO(), "w") as archive, patch("evaluate_s2looking.IMAGE_SHAPE", (8, 8)):
            for folder, pixels in (("Image1", rgb), ("Image2", rgb), ("label1", added), ("label2", removed)):
                self.write_png(archive, f"S2Looking/train/{folder}/valid.png", pixels)
                invalid = pixels.copy()
                if folder == "label1":
                    invalid[0, 0] = (0, 255, 0)
                self.write_png(archive, f"S2Looking/train/{folder}/invalid.png", invalid)
            dataset = S2LookingSplit(archive, "train", 2)
            np.testing.assert_array_equal(dataset.load_pair("valid.png")[2], expected)
            with self.assertRaisesRegex(DatasetError, "label1/invalid.png"):
                dataset.load_pair("invalid.png")

    def test_read_error_identifies_the_corrupt_zip_member(self):
        from evaluate_s2looking import S2LookingSplit
        from levir_data import DatasetError

        buffer = io.BytesIO()
        member = "S2Looking/train/Image1/example.png"
        payload = b"damaged image payload"
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(member, payload)
            archive.writestr("S2Looking/train/Image2/example.png", b"")
            archive.writestr("S2Looking/train/label/example.png", b"")
        damaged = buffer.getvalue().replace(payload, b"X" * len(payload))
        with zipfile.ZipFile(io.BytesIO(damaged)) as archive:
            dataset = S2LookingSplit(archive, "train", 1)
            with self.assertRaisesRegex(DatasetError, "train/Image1/example.png.*CRC"):
                dataset.load_pair("example.png")


if __name__ == "__main__":
    unittest.main()
