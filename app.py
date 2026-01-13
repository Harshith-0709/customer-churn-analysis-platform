import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("📉 Customer Churn Prediction App")
st.write("Upload customer data to predict churn probability.")

st.markdown("---")

uploaded_file = st.file_uploader("Upload CSV file (same format as training data)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head())

        if st.button("Predict Churn"):
            preds = model.predict_proba(data)[:, 1]
            data["Churn_Probability"] = np.round(preds, 3)
            data["Churn_Prediction"] = (preds >= 0.5).astype(int)

            st.subheader("Prediction Results")
            st.dataframe(data.head())

            churn_count = data["Churn_Prediction"].sum()
            st.success(f"Predicted churn customers: {churn_count} / {len(data)}")

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV",
                csv,
                "churn_predictions.csv",
                "text/csv"
            )

    except Exception as e:
        st.error("Error processing file:")
        st.exception(e)

else:
    st.info("Please upload a CSV file to begin.")

st.markdown("---")
st.caption("Built with Streamlit | Customer Churn Prediction Project")
