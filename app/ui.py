"""Streamlit interface for tomato leaf disease detection."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.advisories import advisory_for  # noqa: E402
from src.infer import find_model, load_model, predict  # noqa: E402
from src.leaf_check import assess as leaf_assess  # noqa: E402
from src.settings import CLASS_DISPLAY, CLASS_NAMES  # noqa: E402


st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon=":seedling:",
    layout="centered",
)


def _sidebar() -> None:
    with st.sidebar:
        st.title("Project Info")
        st.markdown(
            "**Tomato Leaf Disease Detector**  \n"
            "Idowu Oluwasemilore Precious  \n"
            "22CD009369 / 2201385  \n"
            "Landmark University, January 2026"
        )
        st.divider()
        st.subheader("Recognised classes")
        for cls in CLASS_NAMES:
            st.write(f"- {CLASS_DISPLAY[cls]}")


def _ensure_model_or_stop() -> None:
    try:
        find_model()
    except FileNotFoundError:
        st.error(
            "No trained model found in `models/`. "
            "Train the model first by running `python -m src.train` in the Anaconda Prompt."
        )
        st.stop()


def main() -> None:
    _sidebar()
    st.title(":seedling: Tomato Leaf Disease Detector")
    st.write(
        "Upload a clear photo of a single tomato leaf. The model classifies it "
        "into one of ten categories (nine diseases plus healthy) and explains the result."
    )

    _ensure_model_or_stop()

    with st.spinner("Loading model..."):
        load_model()

    uploaded = st.file_uploader(
        "Choose a tomato leaf image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    camera_image = None
    with st.expander("Or capture from your camera"):
        camera_image = st.camera_input("Take a photo")

    source = uploaded or camera_image
    if source is None:
        st.info("Awaiting an image...")
        return

    image = Image.open(source)
    st.image(image, caption="Submitted image", use_column_width=True)

    leaf_status = leaf_assess(image)
    if leaf_status["verdict"] == "reject":
        st.error(
            f"This image does not look like a tomato leaf "
            f"(leaf-colour ratio {leaf_status['ratio'] * 100:.1f}%). "
            "The classifier is trained only on tomato leaves and would give "
            "a misleading prediction. Please upload a close-up of a tomato leaf."
        )
        return
    if leaf_status["verdict"] == "warn":
        st.warning(
            f"This image is borderline (leaf-colour ratio "
            f"{leaf_status['ratio'] * 100:.1f}%). Results may be unreliable - "
            "a clearer, well-lit close-up of a single leaf usually works better."
        )

    with st.spinner("Analysing leaf..."):
        result = predict(image, top_k=3)

    confidence_pct = result["confidence"] * 100
    if confidence_pct < 60:
        st.warning(
            f"Low confidence ({confidence_pct:.1f}%). Try a clearer, well-lit, "
            "close-up photo of a single leaf."
        )
    st.subheader(f"Prediction: {result['predicted_label']}")
    st.metric(label="Confidence", value=f"{confidence_pct:.2f}%")

    advisory = advisory_for(result["predicted_class_id"])
    st.markdown("### Advisory")
    st.markdown(f"**Cause** - {advisory['cause']}")
    st.markdown(f"**Symptoms** - {advisory['symptoms']}")
    st.markdown(f"**Recommended action** - {advisory['action']}")

    st.markdown("### Top candidates")
    for entry in result["top_k"]:
        st.write(f"- {entry['label']}: {entry['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    main()
