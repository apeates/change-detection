"""Run the local Turkish TinyCD demonstration."""

from pathlib import Path

import streamlit as st

from baseline_diff import change_counts, metrics_from_counts
from levir_data import DEFAULT_DATA_ROOT
from tinycd_demo import DEMO_SCENES, load_demo_image, load_model, predict_scene
from tinycd_demo_images import comparison_image, mask_png, png_bytes, prediction_overlay


VALIDATION_DIR = DEFAULT_DATA_ROOT / "val"
TEST_RESULT = "LEVIR-CD resmî test sonucu · 128 görüntü çifti · değişim F1 %90,47."


@st.cache_resource(show_spinner=False)
def demo_model():
    model, _ = load_model()
    return model


def clear_scene_result():
    st.session_state.pop("scene_result", None)
    st.session_state.pop("show_reference", None)


def make_scene_result(sample, before, after):
    prediction = predict_scene(demo_model(), before, after)
    scene = Path(sample).stem
    result = {
        "scene": scene,
        "mask_png": mask_png(prediction),
        "overlay_png": png_bytes(prediction_overlay(after, prediction)),
    }
    reference_path = VALIDATION_DIR / "label" / sample
    try:
        reference = load_demo_image(reference_path, "L") >= 128
        result["comparison_png"] = png_bytes(comparison_image(after, prediction, reference, scene))
        result["f1"] = metrics_from_counts(**change_counts(prediction, reference))["F1"]
    except (OSError, ValueError) as error:
        result["reference_error"] = f"Referans açılamadı ({reference_path}). {error}"
    return result


def show_result(result):
    left, right = st.columns([2, 1], vertical_alignment="center")
    compare = left.toggle("Referansla karşılaştır", key="show_reference")
    if "f1" in result:
        score = f"{result['f1'] * 100:.2f}".replace(".", ",")
        right.write(f"**Bu sahnenin F1’i · %{score}**")
    if compare and "reference_error" in result:
        st.error(result["reference_error"])
    image = result.get("comparison_png") if compare else None
    st.image(image or result["overlay_png"], width="stretch")

    mask_col, comparison_col = st.columns(2)
    scene = result["scene"]
    mask_col.download_button(
        "Değişim maskesini indir", result["mask_png"], file_name=f"{scene}_mask.png",
        mime="image/png", on_click="ignore", width="stretch",
    )
    if "comparison_png" in result:
        comparison_col.download_button(
            "Karşılaştırma görselini indir", result["comparison_png"],
            file_name=f"{scene}_karsilastirma.png", mime="image/png",
            on_click="ignore", width="stretch",
        )


def main():
    st.set_page_config(page_title="Bina Değişimi", layout="centered")
    st.title("Bina Değişimi")
    st.caption(TEST_RESULT)
    choice, _ = st.columns([2, 3])
    sample = choice.selectbox("Örnek seç", DEMO_SCENES, format_func=lambda name: Path(name).stem,
                              key="scene", on_change=clear_scene_result)
    paths = [VALIDATION_DIR / side / sample for side in ("A", "B")]
    try:
        before, after = (load_demo_image(path) for path in paths)
    except OSError as error:
        st.error(f"Örnek görüntü açılamadı. {error}")
        return

    for column, title, image in zip(st.columns(2), ("Önce", "Sonra"), (before, after), strict=True):
        column.subheader(title)
        column.image(image, width="stretch")

    if st.button("Değişimi göster", type="primary"):
        try:
            with st.spinner("TinyCD değişim maskesini oluşturuyor…"):
                st.session_state["scene_result"] = make_scene_result(sample, before, after)
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Tahmin oluşturulamadı. {error}")

    result = st.session_state.get("scene_result")
    if result and result["scene"] == Path(sample).stem:
        show_result(result)


if __name__ == "__main__":
    main()
