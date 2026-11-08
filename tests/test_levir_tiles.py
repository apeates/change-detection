import unittest

import numpy as np

from levir_tiles import iter_tiles, orient_tile


class LevirTileTests(unittest.TestCase):
    def test_all_square_orientations_keep_dates_and_mask_aligned(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, :2] = True
        mask[2, 1] = True
        before = np.zeros((5, 5, 3), dtype=np.uint8)
        after = before.copy()
        before[mask, 0] = 255
        after[mask, 2] = 255
        tile = next(iter_tiles(before, after, mask, tile_size=5))

        masks = set()
        for orientation in range(8):
            with self.subTest(orientation=orientation):
                oriented = orient_tile(tile, orientation)
                self.assertEqual(oriented.inputs.shape, tile.inputs.shape)
                self.assertEqual(oriented.mask.shape, tile.mask.shape)
                self.assertTrue(oriented.inputs.flags.c_contiguous)
                self.assertTrue(oriented.mask.flags.c_contiguous)
                np.testing.assert_array_equal(oriented.inputs[0], oriented.mask[0])
                np.testing.assert_array_equal(oriented.inputs[5], oriented.mask[0])
                self.assertEqual(oriented.mask.sum(), mask.sum())
                self.assertEqual(set(np.unique(oriented.mask)), {0, 1})
                masks.add(oriented.mask.tobytes())

        self.assertEqual(len(masks), 8)
        np.testing.assert_array_equal(tile.mask[0], mask)

    def test_tiles_cover_image_and_preserve_alignment(self):
        y, x = np.indices((1024, 1024), dtype=np.uint16)
        before = np.stack((x % 256, y % 256, (x + y) // 8), axis=-1).astype(np.uint8)
        after = 255 - before
        mask = (x >= 254) & (x <= 258) & (y >= 254) & (y <= 258)
        rebuilt = np.empty((6, 1024, 1024), dtype=np.float32)
        rebuilt_mask = np.empty_like(mask)

        tiles = list(iter_tiles(before, after, mask))
        self.assertEqual(len(tiles), 16)
        self.assertEqual([(tile.x, tile.y) for tile in tiles],
                         [(left, top) for top in range(0, 1024, 256) for left in range(0, 1024, 256)])
        for tile in tiles:
            self.assertEqual(tile.inputs.shape, (6, 256, 256))
            self.assertEqual(tile.mask.shape, (1, 256, 256))
            self.assertEqual(tile.inputs.dtype, np.float32)
            self.assertEqual(tile.mask.dtype, np.float32)
            region = np.s_[tile.y:tile.y + 256, tile.x:tile.x + 256]
            rebuilt[:, region[0], region[1]] = tile.inputs
            rebuilt_mask[region] = tile.mask[0]

        np.testing.assert_allclose(rebuilt[:3], before.transpose(2, 0, 1) / 255)
        np.testing.assert_allclose(rebuilt[3:], after.transpose(2, 0, 1) / 255)
        np.testing.assert_array_equal(rebuilt_mask, mask)

    def test_non_divisible_images_do_not_silently_lose_edge_pixels(self):
        image = np.zeros((256, 257, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "multiples of 256"):
            list(iter_tiles(image, image, np.zeros((256, 257), dtype=bool)))

    def test_mask_dimensions_must_match(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "mask must match"):
            list(iter_tiles(image, image, np.zeros((256, 255), dtype=bool)))


if __name__ == "__main__":
    unittest.main()
