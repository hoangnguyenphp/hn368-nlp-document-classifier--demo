import streamlit as st

from config import ModelConfig
from src.inference import Inference


st.set_page_config(
    page_title="Multilingual Document Classifier",
    page_icon="📄",
    layout="centered"
)


@st.cache_resource
def load_predictor():

    return Inference(
        model_dir=ModelConfig.MODEL_REPO
    )


# ==========================================================
# Header
# ==========================================================

st.title("📄 Multilingual Document Classifier")

st.markdown("""
Classify **Vietnamese** and **English** documents into:

- 💰 Economy
- 🏛 Politics
- ⚽ Sports

The model is based on **XLM-RoBERTa** and supports multilingual text classification.
""")


# ==========================================================
# Important Notice
# ==========================================================

st.info(
"""
### ⚠ Important Notice

This demo is hosted on **Streamlit Community Cloud (Free Tier)**.

The AI model is approximately **1.1 GB**, therefore:

- The **first request** after the application wakes up may take **2–3 minutes**.
- During model initialization, Streamlit may temporarily display an error page.
- If that happens, please **refresh the page or come back after 2–3 minutes**.
- Once the model has been loaded, subsequent predictions are much faster.

Thank you for your patience!
"""
)


# ==========================================================
# Example
# ==========================================================

with st.expander("📝 Example Inputs", expanded=False):

    st.markdown("""
**Economy**

- Ngân hàng Nhà nước tăng lãi suất.
- Vietnam GDP reached 7.2%.

**Politics**

- Quốc hội thông qua luật mới.
- The president signed a new bill.

**Sports**

- Manchester United thắng 3-1.
- Lionel Messi scored two goals.
""")


st.divider()


# ==========================================================
# Load model
# ==========================================================

predictor = load_predictor()


# ==========================================================
# Input
# ==========================================================

text = st.text_area(
    "Input Text",
    placeholder="Enter Vietnamese or English text here...",
    height=180
)


# ==========================================================
# Prediction
# ==========================================================

if st.button(
    "Predict",
    type="primary",
    use_container_width=True
):

    if text.strip() == "":

        st.warning(
            "Please enter some text."
        )

    else:

        with st.spinner(
            "Running inference..."
        ):

            result = predictor.predict(text)

        if result["is_unknown"]:

            st.warning(
                "Unknown document"
            )

        else:

            st.success(
                "Prediction Result"
            )

            for category, confidence in zip(
                result["categories"],
                result["confidences"]
            ):

                st.metric(
                    category,
                    f"{confidence:.2%}"
                )

        st.divider()

        st.subheader(
            "Confidence Scores"
        )

        for label, score in result["raw_scores"].items():

            st.write(f"**{label}**")

            st.progress(float(score))

            st.caption(f"{score:.3f}")