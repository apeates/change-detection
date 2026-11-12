"""Compare fixed LEVIR-CD methods under controlled horizontal misregistration."""

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from baseline_diff import change_counts, metrics_from_counts
from evaluate_baseline import CSV_FIELDS
from evaluate_test import (
    EARLY_FUSION, METHOD_LABELS, PIXEL_METHOD, PIXEL_THRESHOLD, SIAMESE,
    SENSITIVITY_EXCLUSIONS, load_frozen_models, write_table,
)
from evaluate_unet import benchmark_checkpoints, predict_mask
from levir_data import DEFAULT_DATA_ROOT, OFFICIAL_COUNTS, load_pair, official_samples


SHIFTS = (0, 1, 2, 3, 4)
OTSU = "rgb_difference_otsu"
CVA = "rgb_cva"


def difference_scores(before, after):
    delta = np.subtract(after, before, dtype=np.float32)
    absolute = np.abs(delta).mean(axis=-1)
    magnitude = np.sqrt(np.square(delta).sum(axis=-1))
    return absolute, np.rint(magnitude).astype(np.int16)


def otsu_threshold(histogram):
    counts = histogram.cumsum()[:-1]
    total = histogram.sum()
    sums = (np.arange(len(histogram)) * histogram).cumsum()
    numerator = np.square(total * sums[:-1] - counts * sums[-1], dtype=np.float64)
    denominator = counts * (total - counts)
    if not np.any(denominator):
        return int(np.flatnonzero(histogram)[0])
    return int(np.argmax(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)))


def cva_threshold(positive, negative):
    true_positive = np.cumsum(positive[::-1])[::-1][1:]
    false_positive = np.cumsum(negative[::-1])[::-1][1:]
    denominator = positive.sum() + true_positive + false_positive
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(len(denominator)), where=denominator > 0)
    return int(f1.argmax())


def calibrate_classical(split, samples):
    otsu = np.zeros(256, dtype=np.int64)
    positive, negative = np.zeros(443, dtype=np.int64), np.zeros(443, dtype=np.int64)
    for sample in samples:
        before, after, mask = load_pair(split, sample)
        absolute, cva = difference_scores(before, after)
        otsu += np.bincount(np.rint(absolute).astype(np.int16).ravel(), minlength=len(otsu))
        positive += np.bincount(cva[mask], minlength=len(positive))
        negative += np.bincount(cva[~mask], minlength=len(negative))
    return {OTSU: otsu_threshold(otsu), CVA: cva_threshold(positive, negative)}


def shift_right(image, pixels):
    if pixels == 0:
        return image
    shifted = np.empty_like(image)
    shifted[:, :pixels] = image[:, :1]
    shifted[:, pixels:] = image[:, :-pixels]
    return shifted


def classical_predictions(before, after, thresholds):
    absolute, cva = difference_scores(before, after)
    return {PIXEL_METHOD: absolute > PIXEL_THRESHOLD,
            OTSU: np.rint(absolute) > thresholds[OTSU], CVA: cva > thresholds[CVA]}


def evaluate_shifts(split, samples, models, selection, thresholds, device):
    methods = (PIXEL_METHOD, OTSU, CVA, *models)
    totals = {pixels: {method: Counter(tp=0, fp=0, fn=0) for method in methods} for pixels in SHIFTS}
    classical = {method: Counter(tp=0, fp=0, fn=0) for method in (OTSU, CVA)}
    valid = np.s_[:, max(SHIFTS):]
    for index, sample in enumerate(samples, 1):
        before, after, reference = load_pair(split, sample)
        for pixels in SHIFTS:
            shifted = shift_right(after, pixels)
            predictions = classical_predictions(before, shifted, thresholds)
            if pixels == 0:
                for method in classical:
                    classical[method].update(change_counts(predictions[method], reference))
            for method, model in models.items():
                predictions[method] = predict_mask(model, before, shifted, reference,
                                                   selection[method]["threshold"], 2, device)
            for method, predicted in predictions.items():
                totals[pixels][method].update(change_counts(predicted[valid], reference[valid]))
        if index % 16 == 0:
            print(f"Shift test {index}/{len(samples)}", flush=True)
    return totals, classical


def write_summary(output, totals, classical, thresholds, resources, labels):
    lines = [
        "# LEVIR-CD kontrollü kıyası", "",
        "Önceki tam görüntü Siamese test referansı %77,64 F1. Kaydırma tablosu her seviyede sol 4 sütunu dışarıda bırakır.",
        "", f"## Klasik yöntemler · {OFFICIAL_COUNTS['test']} test çifti", "",
        "| Yöntem | Eşik | F1 | IoU | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, counts in classical.items():
        metrics = metrics_from_counts(**counts)
        lines.append(f"| {labels[method]} | {thresholds[method]} | {metrics['F1']:.2%} | {metrics['IoU']:.2%} | "
                     f"{metrics['Precision']:.2%} | {metrics['Recall']:.2%} |")
    lines.extend([
        "", "## Sağa kaydırma · Ortak geçerli piksel alanı", "",
        "| Yöntem | 0 piksel F1 | 4 piksel F1 | Fark (yüzde puan) |",
        "|---|---:|---:|---:|",
    ])
    for method, counts in totals[0].items():
        aligned = metrics_from_counts(**counts)["F1"]
        shifted = metrics_from_counts(**totals[4][method])["F1"]
        lines.append(f"| {labels[method]} | {aligned:.2%} | {shifted:.2%} | {(shifted - aligned) * 100:+.2f} |")
    lines.extend([
        "", f"## Kaynak ölçümü · {resources['device']}", "",
        f"{resources['sample']} çiftinin {resources['tiles']} parçası, batch {resources['batch_size']}, "
        f"{resources['warmup_passes']} ısınma ve {resources['timed_passes']} zamanlı geçiş. "
        "Süreye veri yükleme ve maske birleştirme dahil edilmedi.", "",
        "| Model | Parametre | Medyan ileri geçiş (ms/çift) | Tepe GPU belleği (MiB) |",
        "|---|---:|---:|---:|",
    ])
    models = resources["models"]
    for method, row in models.items():
        peak = row["peak_allocated_bytes"]
        peak_mib = f"{peak / 2**20:.2f}" if peak is not None else "—"
        lines.append(f"| {labels[method]} | {row['parameters']:,} | {row['median_forward_ms_per_pair']:.2f} | {peak_mib} |")
    siamese, early = models[SIAMESE], models[EARLY_FUSION]
    time_delta = siamese["median_forward_ms_per_pair"] - early["median_forward_ms_per_pair"]
    lines.extend(["", f"Siamese − Early Fusion medyan süre farkı **{time_delta:+.2f} ms/çift**."])
    if siamese["peak_allocated_bytes"] is not None and early["peak_allocated_bytes"] is not None:
        peak_delta = (siamese["peak_allocated_bytes"] - early["peak_allocated_bytes"]) / 2**20
        lines.append(f"Siamese − Early Fusion tepe GPU belleği farkı **{peak_delta:+.2f} MiB**.")
    else:
        lines.append("GPU belleği yalnızca CUDA çalıştırmasında ölçülür.")
    lines.extend([
        "", "## Önceki BIT deneyi", "",
        "BIT eğitimi 12 tamamlanmış tur ve 42.720 güncellemede durduruldu. En iyi doğrulama F1'i "
        "1. turda %63,05, 12. turdaki doğrulama F1'i %34,24. Her iki skor da 0.50 eşiğinde ölçüldü.",
        "", "Resmî test ayrımında eğitim alanıyla kısmen örtüşen iki çift bulunur "
        f"({', '.join(sorted(SENSITIVITY_EXCLUSIONS))}).",
    ])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(output, totals, classical, thresholds, resources, selection):
    labels = {**METHOD_LABELS, OTSU: "RGB difference + Otsu", CVA: "RGB CVA"}
    limits = {method: choice["threshold"] for method, choice in selection.items()} | thresholds
    rows = [{"shift_px": pixels, "method": method, "threshold": limits[method],
             **counts, **metrics_from_counts(**counts)}
            for pixels, methods in totals.items() for method, counts in methods.items()]
    write_table(output / "robustness.csv", ("shift_px", "method", "threshold", *CSV_FIELDS[1:]), rows)
    write_table(output / "classical_test.csv", ("method", "threshold", *CSV_FIELDS[1:]),
                ({"method": method, "threshold": thresholds[method], **counts, **metrics_from_counts(**counts)}
                 for method, counts in classical.items()))
    (output / "resources.json").write_text(json.dumps(resources, indent=2) + "\n")
    (output / "protocol.json").write_text(json.dumps({
        "thresholds": limits, "validation_pairs": OFFICIAL_COUNTS["val"],
        "test_pairs": OFFICIAL_COUNTS["test"], "shifts_px": SHIFTS,
        "common_valid_region": "All rows; columns 4 onward for every shift, including zero.",
        "otsu": "Single threshold from the pooled, rounded validation RGB difference histogram.",
        "cva": "Rounded Euclidean RGB difference; validation F1 over all integer thresholds, ties choose lower.",
        "known_partial_train_test_overlaps": sorted(SENSITIVITY_EXCLUSIONS),
        "selection": selection,
    }, indent=2) + "\n")
    fig, axis = plt.subplots(layout="constrained")
    for method in totals[0]:
        axis.plot(SHIFTS, [row["F1"] for row in rows if row["method"] == method], marker="o", label=labels[method])
    axis.set(xlabel="Horizontal T2 shift (pixels)", ylabel="Change F1", xticks=SHIFTS, ylim=(0, 1))
    axis.legend()
    fig.savefig(output / "robustness.png", dpi=140)
    plt.close(fig)
    write_summary(output, totals, classical, thresholds, resources, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--bit-run", type=Path, help="Include a BIT run with a frozen validation selection.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output already exists: {args.output_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Enable a GPU runtime.")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    val = args.data_root / "val"
    test = args.data_root / "test"
    val_samples = official_samples(val)
    samples = official_samples(test)
    models, selection = load_frozen_models(args.runs_root, args.device, bit_run=args.bit_run)
    print(f"Calibrating Otsu and CVA on {len(val_samples)} validation pairs.", flush=True)
    thresholds = calibrate_classical(val, val_samples)
    totals, classical = evaluate_shifts(test, samples, models, selection, thresholds, args.device)
    del models
    checkpoints = {name: Path(choice["checkpoint"]) for name, choice in selection.items() if "checkpoint" in choice}
    resources = benchmark_checkpoints(checkpoints, val, 2, args.device)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as temporary:
        staged = Path(temporary) / "report"
        staged.mkdir()
        write_reports(staged, totals, classical, thresholds, resources, selection)
        staged.rename(args.output_dir)
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
