import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, feature_cols

model, feature_columns = load_assets()

st.title("📉 Customer Churn Prediction App")
st.write("Upload customer CSV data (raw format) to predict churn probability.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head())

        if st.button("Predict Churn"):

            # Drop ID if present
            if "customerID" in data.columns:
                data = data.drop(columns=["customerID"])

            # One-hot encode
            data_encoded = pd.get_dummies(data)

            # Align with training columns
            for col in feature_columns:
                if col not in data_encoded.columns:
                    data_encoded[col] = 0

            data_encoded = data_encoded[feature_columns]

            preds = model.predict_proba(data_encoded)[:, 1]

            results = data.copy()
            results["Churn_Probability"] = np.round(preds, 3)
            results["Churn_Prediction"] = (preds >= 0.5).astype(int)

            st.subheader("Prediction Results")
            st.dataframe(results.head())

            churn_count = results["Churn_Prediction"].sum()
            st.success(f"Predicted churn customers: {churn_count} / {len(results)}")

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV",
                csv,
                "churn_predictions.csv",
                "text/csv"
            )

    except Exception as e:
        st.error("Error processing file")
        st.exception(e)

else:
    st.info("Please upload a CSV file to begin.")
