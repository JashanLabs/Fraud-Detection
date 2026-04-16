
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# LOAD MODELS & FEATURES
# -----------------------------
rf = joblib.load("rf_model.pkl")
iso = joblib.load("iso_model.pkl")
feature_order = joblib.load("features.pkl")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 Fraud Detection & Investigation Dashboard")
st.markdown("Upload a transaction dataset to detect suspicious activity using a hybrid ML system.")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df_original = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data")
    st.dataframe(df_original.head())

    # -----------------------------
    # VALIDATE COLUMNS
    # -----------------------------
    missing_cols = set(feature_order) - set(df_original.columns)
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.stop()

    # -----------------------------
    # PREPARE MODEL INPUT
    # -----------------------------
    X_input = df_original[feature_order].copy()

    # -----------------------------
    # PREDICTIONS
    # -----------------------------
    probs = rf.predict_proba(X_input)[:, 1]
    df_original['Risk Score'] = probs * 100

    anomaly_preds = iso.predict(X_input)
    df_original['Anomaly'] = anomaly_preds

    # -----------------------------
    # HYBRID DECISION LOGIC
    # -----------------------------
    df_original['Final Decision'] = df_original.apply(
        lambda row: "Investigate"
        if row['Risk Score'] > 40 or (row['Risk Score'] > 20 and row['Anomaly'] == -1)
        else "Safe",
        axis=1
    )

    # -----------------------------
    # METRICS
    # -----------------------------
    fraud_count = (df_original['Final Decision'] == "Investigate").sum()
    avg_risk = df_original['Risk Score'].mean()

    col1, col2 = st.columns(2)
    col1.metric("🚨 Transactions to Investigate", fraud_count)
    col2.metric("📊 Average Risk Score", round(avg_risk, 2))

    # -----------------------------
    # TOP SUSPICIOUS
    # -----------------------------
    st.subheader("🔍 Top Suspicious Transactions")

    top_suspicious = df_original[df_original['Final Decision'] == "Investigate"] \
        .sort_values(by='Risk Score', ascending=False) \
        .head(10)

    st.dataframe(top_suspicious)

    # -----------------------------
    # FULL TABLE
    # -----------------------------
    st.subheader("📋 All Transactions")
    st.dataframe(df_original)

    df_original['Risk Level'] = df_original['Risk Score'].apply(
    lambda x: "High 🔴" if x > 70 else "Medium 🟡" if x > 40 else "Low 🟢")
    st.subheader("📊 Risk Breakdown")

    st.write(df_original['Risk Level'].value_counts())
    # -----------------------------
    # RISK DISTRIBUTION
    # -----------------------------
    st.subheader("📊 Risk Score Distribution")

    fig, ax = plt.subplots()
    ax.hist(df_original['Risk Score'], bins=50)
    ax.set_title("Risk Score Distribution")

    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin.")
