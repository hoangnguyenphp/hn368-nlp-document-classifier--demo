import streamlit as st

from config import ModelConfig
from src.inference import Inference


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multilingual Document Classifier",
    page_icon="📄",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_predictor():

    return Inference(
        model_dir=ModelConfig.MODEL_REPO
    )


predictor = load_predictor()


# ============================================================
# HEADER
# ============================================================

st.title("📄 Multilingual Document Classifier")

st.markdown("""
This demo uses a **fine-tuned XLM-RoBERTa Transformer model**
to classify **English** and **Vietnamese** documents.

### Supported Categories

- 💰 Economy
- 🏛 Politics
- ⚽ Sports

The model supports **multi-label classification**, meaning
a document can belong to multiple categories simultaneously.
""")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.markdown("""
**Model**

XLM-RoBERTa Base

**Task**

Multilingual Text Classification

**Languages**

- 🇻🇳 Vietnamese
- 🇺🇸 English

**Framework**

- PyTorch
- Hugging Face Transformers

**Deployment**

- Streamlit Community Cloud

---
Built by **Hoang Nguyen**
""")

    st.divider()

    st.subheader("Example Inputs")

    example = st.selectbox(

        "Choose an example",

        [
            "Chính phủ công bố gói hỗ trợ kinh tế 100000 nghìn tỷ đồng",

            "Quốc hội thông qua luật mới.",

            "Manchester United won the championship.",

            "Chính phủ công bố gói hỗ trợ kinh tế mới.",

            "Hôm nay trời đẹp quá."

        ]

    )

    if st.button("Use Example"):

        st.session_state["example_text"] = example


# ============================================================
# INPUT
# ============================================================

default_text = st.session_state.get(
    "example_text",
    ""
)

text = st.text_area(

    "Input Document",

    value=default_text,

    height=200,

    placeholder="""
Example:

Chính phủ công bố gói hỗ trợ kinh tế 100000 nghìn tỷ đồng.

or

Manchester United won the Premier League title.

"""
)


# ============================================================
# PREDICT
# ============================================================

if st.button(
    "🚀 Predict",
    type="primary",
    use_container_width=True
):

    if text.strip() == "":

        st.warning("Please enter a document.")

    else:

        with st.spinner("Running inference..."):

            result = predictor.predict(text)

        st.divider()

        st.subheader("Prediction Result")

        if result["is_unknown"]:

            st.warning(
                "⚠️ This document does not belong to any known category."
            )

        else:

            for category, confidence in zip(

                result["categories"],

                result["confidences"]

            ):

                st.success(category)

                st.progress(float(confidence))

                st.write(
                    f"Confidence: **{confidence:.2%}**"
                )

        st.divider()

        st.subheader("Confidence Scores")

        raw_scores = result["raw_scores"]

        for label, score in raw_scores.items():

            st.write(f"**{label}**")

            st.progress(float(score))

            st.caption(f"{score:.3f}")