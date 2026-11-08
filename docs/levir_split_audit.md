# LEVIR-CD split audit

The local dataset contains 445 train, 64 validation, and 128 test image pairs. Each filename has both RGB images and a reference mask. All 637 filenames match the [publisher's footprint metadata](https://github.com/justchenhao/STANet/tree/master/LEVIR_CD_spatial_info).

## Method

We matched filenames to the coordinate rectangles in `LEVIR_CD_name_coords.geojson` and compared every pair of rectangles from different splits. Two rectangles overlap when the intersection has positive width and height. Boundary contacts have zero area. The overlap percentages use the smaller rectangle as the denominator.

Source file SHA-256: `1f0d80046d85658be5091f2be0ffe746491889ee07524e8c4c9585b4aded0d64`.

## Results

| Splits | Duplicate filenames | Overlapping pairs | Boundary contacts |
|---|---:|---:|---:|
| Train and validation | 0 | 0 | 252 |
| Train and test | 0 | 2 | 473 |
| Validation and test | 0 | 0 | 68 |

- `train_2.png` overlaps `test_9.png` by about 25%.
- `train_87.png` overlaps `test_19.png` by about 6.25%.

Use the standard 128-pair test split for the benchmark. Also report a sensitivity result for the 126 pairs that remain after excluding `test_9.png` and `test_19.png`. Use the same checkpoints, thresholds, and predictions for both results. Many footprints touch, so the 126-pair result measures the effect of these two overlaps. An independent geographic evaluation would require a separate holdout region.
