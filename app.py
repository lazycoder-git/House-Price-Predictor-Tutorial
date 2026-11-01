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

# Sidebar controls for error percentage / uncertainty margin
with st.sidebar:
    st.header("⚙️ Valuation Settings")
    st.write(
        "By default, the price range is derived from the model's test Mean Absolute Error "
        f"(MAE: ±${package.get('metrics', {}).get('test_mae', 25415.0):,.0f})."
    )
    custom_margin = st.checkbox("Set custom error percentage", value=False)
    if custom_margin:
        chosen_pct = st.slider(
            "Error percentage (±%)",
            min_value=5.0,
            max_value=30.0,
            value=14.0,
            step=0.5,
            help="Specify a custom percentage margin for the estimated price range."
        )
    else:
        chosen_pct = None

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
        test_mae = float(package.get("metrics", {}).get("test_mae", 25415.06))

        if chosen_pct is not None:
            pct = chosen_pct
            margin = prediction * (pct / 100.0)
        else:
            margin = test_mae
            pct = (margin / prediction * 100.0) if prediction > 0 else 14.0

        lower_price = max(0.0, prediction - margin)
        upper_price = prediction + margin

        st.subheader("Price Prediction")
        st.success(
            f"### 🏷️ Estimated Price Range: **${lower_price:,.0f} – ${upper_price:,.0f}**"
        )

        col_low, col_mid, col_high = st.columns(3)
        col_low.metric(
            label=f"Lower Bound (-{pct:.1f}%)",
            value=f"${lower_price:,.0f}",
            delta=f"-${margin:,.0f}",
            delta_color="off"
        )
        col_mid.metric(
            label="Expected Estimate",
            value=f"${prediction:,.0f}"
        )
        col_high.metric(
            label=f"Upper Bound (+{pct:.1f}%)",
            value=f"${upper_price:,.0f}",
            delta=f"+${margin:,.0f}",
            delta_color="off"
        )

        st.caption(
            f"Range reflects an error margin of approximately ±{pct:.1f}% "
            f"({'custom percentage' if chosen_pct is not None else f'based on model test MAE of ±${test_mae:,.0f}'}). "
            "This is a statistical estimate, not a formal appraisal."
        )

        with st.expander("Show model details"):
            st.write("Features used:", features)
            if package.get("metrics"):
                st.write("Saved test metrics:", package["metrics"])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("Linear Regression • Six features • Model loaded from Joblib")
