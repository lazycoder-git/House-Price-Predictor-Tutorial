import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "house_model_6_features.pkl"

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def load_package():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place house_model_6_features.pkl in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)

st.title("🏠 House Price Predictor")
st.write(
    "Enter the property's details below. The saved six-feature Linear Regression "
    "model will estimate its sale price."
)

try:
    package = load_package()
except Exception as exc:
    st.error(str(exc))
    st.stop()

model = package["model"]
features = package["features"]
medians = package.get("medians", {})
lower_bound = package.get("lower_bound", {})
upper_bound = package.get("upper_bound", {})

# Friendly labels and sensible input controls for the six project features.
feature_config = {
    "Gr Liv Area": {
        "label": "Above-ground living area (square feet)",
        "min_value": 100,
        "max_value": 10000,
        "value": 1500,
        "step": 50
    },
    "Overall Qual": {
        "label": "Overall quality (1–10)",
        "min_value": 1,
        "max_value": 10,
        "value": 6,
        "step": 1
    },
    "Total Bsmt SF": {
        "label": "Total basement area (square feet)",
        "min_value": 0,
        "max_value": 5000,
        "value": 800,
        "step": 25
    },
    "Garage Area": {
        "label": "Garage area (square feet)",
        "min_value": 0,
        "max_value": 2000,
        "value": 400,
        "step": 25
    },
    "Year Built": {
        "label": "Year built",
        "min_value": 1800,
        "max_value": 2026,
        "value": 2000,
        "step": 1
    },
    "Full Bath": {
        "label": "Number of full bathrooms",
        "min_value": 0,
        "max_value": 8,
        "value": 2,
        "step": 1
    }
}

with st.form("house_details"):
    st.subheader("Property details")
    inputs = {}

    col1, col2 = st.columns(2)
    for index, feature in enumerate(features):
        cfg = feature_config.get(feature)
        if cfg is None:
            # Fallback for any unexpected feature name in the saved artifact.
            default = float(medians.get(feature, 0))
            inputs[feature] = st.number_input(
                feature,
                value=default,
                key=f"feature_{feature}"
            )
            continue

        container = col1 if index % 2 == 0 else col2
        with container:
            inputs[feature] = st.number_input(
                cfg["label"],
                min_value=cfg["min_value"],
                max_value=cfg["max_value"],
                value=cfg["value"],
                step=cfg["step"],
                key=f"feature_{feature}"
            )

    submitted = st.form_submit_button("Predict sale price", type="primary")

if submitted:
    input_df = pd.DataFrame([[inputs[feature] for feature in features]], columns=features)

    # Apply the same Gr Liv Area IQR clipping used during training, if bounds were saved.
    if "Gr Liv Area" in input_df.columns:
        lower = lower_bound.get("Gr Liv Area") if isinstance(lower_bound, dict) else None
        upper = upper_bound.get("Gr Liv Area") if isinstance(upper_bound, dict) else None
        if lower is not None and upper is not None:
            original_area = input_df.loc[0, "Gr Liv Area"]
            input_df.loc[:, "Gr Liv Area"] = input_df["Gr Liv Area"].clip(
                lower=lower, upper=upper
            )
            if original_area != input_df.loc[0, "Gr Liv Area"]:
                st.info(
                    "Living area was clipped to the IQR limits used during training "
                    "before prediction."
                )

    try:
        prediction = float(model.predict(input_df[features])[0])
        st.success(f"Estimated sale price: ${prediction:,.0f}")
        st.caption(
            "This is a model estimate, not a formal appraisal. Accuracy depends on "
            "the training data and the property's similarity to it."
        )

        with st.expander("Show model details"):
            st.write("Features used:", features)
            if package.get("metrics"):
                st.write("Saved test metrics:", package["metrics"])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("Linear Regression • Six features • Model loaded from Joblib")
