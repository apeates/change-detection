"""Run the local Turkish TinyCD demonstration."""

from pathlib import Path
from time import perf_counter

import streamlit as st

from baseline_diff import change_counts, metrics_from_counts, predict_change
from levir_data import DEFAULT_DATA_ROOT, OFFICIAL_COUNTS
from tinycd_demo import (
    DEMO_SCENES, TEST_DEMO_SCENES, THRESHOLD, count_regions,
    load_demo_image, load_model, predict_probabilities,
)
from tinycd_demo_images import (
    comparison_image, error_overlay, mask_png, png_bytes,
    prediction_overlay, probability_heatmap,
)


TEST_RESULT = (
    "LEVIR-CD resmî test sonucu · 128 görüntü çifti · "
    "F1 %90,47 · IoU %82,60 · Precision %92,28 · Recall %88,73"
)
PIXEL_BASELINE_THRESHOLD = 30
SENSITIVITY_OVERLAPS = frozenset(("test_9.png", "test_19.png"))
SPLIT_LABELS = {
    "val": f"Doğrulama ({OFFICIAL_COUNTS['val']})",
    "test": f"Test ({OFFICIAL_COUNTS['test']})",
}
SCENE_NOTES = {
    "val_56.png": "Temiz örnek",
    "val_39.png": "Zorlu / yoğun",
    "test_102.png": "Yüksek doğruluk",
    "test_18.png": "Hata analizi",
    "test_108.png": "Değişim yok (0 yanlış alarm)",
}
VIEW_MODES = (
    "Tahmin Katmanı",
    "Hata Analizi",
    "Olasılık Haritası",
    "Piksel Farkıyla Kıyas",
)


def fmt_pct(value):
    return f"%{value * 100:.2f}".replace(".", ",")


def fmt_int(value):
    return f"{int(value):,}".replace(",", ".")


def fmt_float(value):
    return f"{value:.2f}".replace(".", ",")


def available_splits():
    splits = []
    for split in ("val", "test"):
        directory = DEFAULT_DATA_ROOT / split
        if (directory / "A").is_dir() and (directory / "B").is_dir():
            splits.append(split)
    return tuple(splits) or ("val",)


def scenes_for_split(split):
    featured = DEMO_SCENES if split == "val" else TEST_DEMO_SCENES
    all_scenes = tuple(f"{split}_{index}.png" for index in range(1, OFFICIAL_COUNTS[split] + 1))
    directory = DEFAULT_DATA_ROOT / split / "A"
    existing = {name for name in all_scenes if (directory / name).is_file()}
    ordered = [name for name in featured if name in existing]
    ordered.extend(name for name in all_scenes if name in existing and name not in ordered)
    return tuple(ordered) or featured


def format_scene_name(name):
    stem = Path(name).stem
    note = SCENE_NOTES.get(name)
    return f"{stem} · {note}" if note else stem


@st.cache_resource(show_spinner=False)
def demo_model():
    model, _ = load_model()
    return model


def clear_scene_result():
    st.session_state.pop("scene_result", None)
    st.session_state.pop("threshold", None)


def make_scene_result(split, sample, before, after):
    started = perf_counter()
    probabilities = predict_probabilities(demo_model(), before, after)
    elapsed = perf_counter() - started
    scene = Path(sample).stem
    result = {
        "key": f"{split}/{sample}",
        "split": split,
        "scene": scene,
        "sample": sample,
        "probabilities": probabilities,
        "elapsed": elapsed,
        "baseline_prediction": predict_change(before, after, PIXEL_BASELINE_THRESHOLD),
    }
    reference_path = DEFAULT_DATA_ROOT / split / "label" / sample
    try:
        result["reference"] = load_demo_image(reference_path, "L") >= 128
    except (OSError, ValueError) as error:
        result["reference_error"] = f"Referans açılamadı ({reference_path}). {error}"
    return result


def render_metrics(prediction, reference, regions, elapsed):
    changed_pct = float(prediction.mean())
    col1, col2, col3, col4 = st.columns(4)

    if reference is not None:
        counts = change_counts(prediction, reference)
        metrics = metrics_from_counts(**counts)
        if not reference.any():
            fp = counts["fp"]
            col1.metric("Sahne Durumu", "Değişim yok",
                        "0 yanlış alarm" if fp == 0 else f"{fmt_int(fp)} px yanlış alarm",
                        delta_color="normal" if fp == 0 else "inverse",
                        delta_arrow="off", border=True)
            col2.metric("Referans Değişim", "0 px", f"Tahmin: {fmt_int(regions['total_pixels'])} px",
                        delta_color="off", delta_arrow="off", border=True)
        else:
            col1.metric("Sahne F1", fmt_pct(metrics["F1"]), f"IoU {fmt_pct(metrics['IoU'])}",
                        delta_color="off", delta_arrow="off", border=True)
            col2.metric("Precision / Recall",
                        f"{fmt_pct(metrics['Precision'])} / {fmt_pct(metrics['Recall'])}",
                        f"FP {fmt_int(counts['fp'])} · FN {fmt_int(counts['fn'])} px",
                        delta_color="off", delta_arrow="off", border=True)
    else:
        col1.metric("Sahne F1", "—", "Referans yok", delta_color="off", delta_arrow="off", border=True)
        col2.metric("Precision / Recall", "—", "Referans yok", delta_color="off", delta_arrow="off", border=True)

    region_label = f"{fmt_int(regions['count'])} bölge"
    col3.metric("Tespit Edilen Değişim", region_label,
                f"{fmt_pct(changed_pct)} alan ({fmt_int(regions['total_pixels'])} px)",
                delta_color="off", delta_arrow="off", border=True)
    col4.metric("Çıkarım Süresi", f"{fmt_float(elapsed)} sn", "CPU · float32 · 16 parça",
                delta_color="off", delta_arrow="off", border=True)


def show_centered_image(image_bytes, caption):
    _, center, _ = st.columns([0.55, 2.3, 0.55])
    center.image(image_bytes, width="stretch")
    center.caption(caption)


def show_result(result, after):
    st.divider()
    ctrl_mode, ctrl_thresh, ctrl_alpha = st.columns([2.4, 1.3, 1.1], vertical_alignment="bottom")
    mode = ctrl_mode.segmented_control(
        "Görünüm", VIEW_MODES, default=VIEW_MODES[0], required=True, key="view_mode",
    )
    threshold = ctrl_thresh.slider(
        "Karar eşiği (kilitli: 0,50)", 0.05, 0.95, THRESHOLD, 0.05, key="threshold",
    )
    alpha = ctrl_alpha.slider("Katman opaklığı", 0.20, 0.90, 0.55, 0.05, key="alpha")

    probabilities = result["probabilities"]
    reference = result.get("reference")
    prediction = probabilities > threshold
    regions = count_regions(prediction, min_pixels=16)

    render_metrics(prediction, reference, regions, result["elapsed"])
    scene = result["scene"]

    if mode == "Tahmin Katmanı":
        overlay = png_bytes(prediction_overlay(after, prediction, alpha=alpha))
        show_centered_image(
            overlay,
            f"Mavi katman: TinyCD değişim tahmini (p > {fmt_float(threshold)}).",
        )
    elif mode == "Hata Analizi":
        if reference is None:
            st.error(result.get("reference_error", "Referans maske bulunamadı."))
            overlay = png_bytes(prediction_overlay(after, prediction, alpha=alpha))
            show_centered_image(overlay, "Referans olmadan tahmin katmanı gösteriliyor.")
        else:
            counts = change_counts(prediction, reference)
            comp = png_bytes(comparison_image(after, prediction, reference, scene, alpha=alpha))
            show_centered_image(
                comp,
                f"Doğru (TP): {fmt_int(counts['tp'])} px · "
                f"Kaçırılan (FN): {fmt_int(counts['fn'])} px · "
                f"Fazladan (FP): {fmt_int(counts['fp'])} px",
            )
    elif mode == "Olasılık Haritası":
        heat = png_bytes(probability_heatmap(after, probabilities, threshold=threshold,
                                             alpha=min(0.95, max(alpha, 0.45))))
        show_centered_image(
            heat,
            "Koyu mor → turuncu → parlak sarı artan değişim olasılığını gösterir. "
            "Mavi çizgi aktif karar eşiğini işaretler.",
        )
    else:
        baseline_pred = result["baseline_prediction"]
        left, right = st.columns(2)
        if reference is not None:
            base_counts = change_counts(baseline_pred, reference)
            tiny_counts = change_counts(prediction, reference)
            base_metrics = metrics_from_counts(**base_counts)
            tiny_metrics = metrics_from_counts(**tiny_counts)
            base_img = png_bytes(error_overlay(after, baseline_pred, reference, alpha=alpha))
            tiny_img = png_bytes(error_overlay(after, prediction, reference, alpha=alpha))
            if reference.any():
                base_title = f"Klasik Piksel Farkı (eşik 30) · F1 {fmt_pct(base_metrics['F1'])}"
                tiny_title = f"TinyCD (p > {fmt_float(threshold)}) · F1 {fmt_pct(tiny_metrics['F1'])}"
            else:
                base_title = f"Klasik Piksel Farkı (eşik 30) · FP {fmt_int(base_counts['fp'])} px"
                tiny_title = f"TinyCD (p > {fmt_float(threshold)}) · FP {fmt_int(tiny_counts['fp'])} px"
            left.subheader(base_title)
            left.image(base_img, width="stretch")
            left.caption(f"TP {fmt_int(base_counts['tp'])} · FN {fmt_int(base_counts['fn'])} · FP {fmt_int(base_counts['fp'])} px")
            right.subheader(tiny_title)
            right.image(tiny_img, width="stretch")
            right.caption(f"TP {fmt_int(tiny_counts['tp'])} · FN {fmt_int(tiny_counts['fn'])} · FP {fmt_int(tiny_counts['fp'])} px")
            st.caption("Renkler · Yeşil: doğru yakalanan · Mor: kaçırılan · Turuncu: fazladan işaretlenen")
        else:
            left.subheader("Klasik Piksel Farkı (eşik 30)")
            left.image(png_bytes(prediction_overlay(after, baseline_pred, alpha=alpha)), width="stretch")
            right.subheader(f"TinyCD (p > {fmt_float(threshold)})")
            right.image(png_bytes(prediction_overlay(after, prediction, alpha=alpha)), width="stretch")

    _, dl_left, dl_right, _ = st.columns([0.55, 1.15, 1.15, 0.55])
    dl_left.download_button(
        "Değişim maskesini indir (.png)", mask_png(prediction),
        file_name=f"{scene}_mask.png", mime="image/png",
        on_click="ignore", width="stretch",
    )
    if reference is not None:
        dl_right.download_button(
            "Karşılaştırma görselini indir (.png)",
            png_bytes(comparison_image(after, prediction, reference, scene, alpha=alpha)),
            file_name=f"{scene}_karsilastirma.png", mime="image/png",
            on_click="ignore", width="stretch",
        )


def main():
    st.set_page_config(page_title="Bina Değişimi · TinyCD", layout="wide")
    st.title("Bina Değişimi · TinyCD")
    st.caption(TEST_RESULT)

    splits = available_splits()
    col_split, col_scene, col_action = st.columns([1.4, 2.2, 1.2], vertical_alignment="bottom")
    split = col_split.segmented_control(
        "Veri bölümü", splits, default=splits[0], required=True,
        format_func=lambda key: SPLIT_LABELS[key],
        key="split", on_change=clear_scene_result,
    )
    scenes = scenes_for_split(split)
    sample = col_scene.selectbox(
        "Örnek seç", scenes, format_func=format_scene_name,
        key=f"scene_{split}", on_change=clear_scene_result,
    )
    run_clicked = col_action.button("Değişimi göster", type="primary", width="stretch")

    if sample in SENSITIVITY_OVERLAPS:
        st.caption(
            "Not: Bu test çifti eğitim alanıyla kısmen örtüşür "
            "(126 çiftlik duyarlılık analizinde dışarıda tutulur: F1 %90,50)."
        )

    split_dir = DEFAULT_DATA_ROOT / split
    try:
        before, after = (load_demo_image(split_dir / side / sample) for side in ("A", "B"))
    except OSError as error:
        st.error(f"Örnek görüntü açılamadı. {error}")
        return

    for column, title, image in zip(st.columns(2), ("Önce (T1)", "Sonra (T2)"), (before, after), strict=True):
        column.subheader(title)
        column.image(image, width="stretch")

    if run_clicked:
        try:
            with st.spinner("TinyCD değişim haritasını hesaplıyor…"):
                st.session_state["scene_result"] = make_scene_result(split, sample, before, after)
                st.session_state.pop("threshold", None)
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Tahmin oluşturulamadı. {error}")

    result = st.session_state.get("scene_result")
    if result and result["key"] == f"{split}/{sample}":
        show_result(result, after)


if __name__ == "__main__":
    main()
