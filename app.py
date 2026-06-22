"""
app.py
------
Streamlit web app for the Customer Churn Prediction project.

Features:
  - Enter a new customer's details + their feedback text -> get churn prediction
  - NLP sentiment analysis (VADER) on the feedback text feeds into the model
  - Risk segmentation (Low / Medium / High)
  - SHAP-based explanation: WHY this customer is likely to churn
  - Automatic retention recommendation based on top churn drivers
  - Dashboard tab with overall dataset insights (charts)
  - "Impact of NLP feature" tab showing before/after model performance

Run with:  streamlit run app.py   (from the project root folder)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

from preprocess import feature_engineer
from sentiment import get_sentiment_scores, get_sentiment_label

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

# ---------------------------------------------------------------
# Load trained artifacts (cached so they load only once)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    model_name = joblib.load("models/best_model_name.pkl")
    explainer = joblib.load("models/explainer.pkl")
    return model, scaler, encoders, feature_names, model_name, explainer


@st.cache_data
def load_raw_df():
    return pd.read_csv("data/Telco-Customer-Churn-With-Feedback.csv")


@st.cache_data
def load_sentiment_impact():
    with open("outputs/sentiment_impact.json") as f:
        return json.load(f)


model, scaler, encoders, feature_names, model_name, explainer = load_artifacts()
raw_df = load_raw_df()

st.title("📉 Customer Churn Prediction System")
st.caption(f"Powered by **{model_name}** + NLP Sentiment Analysis  |  BTech Final Year Project")

tab1, tab2, tab3 = st.tabs(["🔮 Predict Churn", "📊 Dashboard", "🧠 NLP Feature Impact"])

# =================================================================
# TAB 1 : Single customer prediction
# =================================================================
with tab1:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * max(tenure, 1)))

    st.markdown("#### 💬 Customer Feedback (NLP Input)")
    feedback_text = st.text_area(
        "Paste / type the customer's feedback, review, or support-call note",
        placeholder="e.g. The internet keeps disconnecting and support never resolves my tickets. Very frustrated.",
        height=100,
    )

    predict_btn = st.button("🔍 Predict Churn", type="primary", use_container_width=True)

    if predict_btn:
        if not feedback_text.strip():
            st.warning("Please enter some feedback text — it's a key input for this model.")
            st.stop()

        # ---- NLP step: turn the feedback text into sentiment features ----
        scores = get_sentiment_scores(feedback_text)
        sentiment_label = get_sentiment_label(scores["compound"])

        # Build a single-row dataframe matching the training schema
        input_dict = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
            "sentiment_compound": scores["compound"], "sentiment_pos": scores["pos"],
            "sentiment_neg": scores["neg"], "sentiment_neu": scores["neu"],
        }
        input_df = pd.DataFrame([input_dict])
        input_df = feature_engineer(input_df)

        # Encode using the SAME encoders fitted during training
        for col, le in encoders.items():
            if col in input_df.columns:
                val = str(input_df[col].iloc[0])
                if val in le.classes_:
                    input_df[col] = le.transform([val])
                else:
                    input_df[col] = 0  # unseen category fallback

        input_df = input_df[feature_names]  # ensure correct column order

        if model_name == "Logistic Regression":
            X_for_pred = scaler.transform(input_df)
        else:
            X_for_pred = input_df

        pred = model.predict(X_for_pred)[0]
        prob = model.predict_proba(X_for_pred)[0][1]

        # ---- Risk segmentation ----
        if prob < 0.35:
            risk_label = "🟢 Low Risk"
        elif prob < 0.65:
            risk_label = "🟡 Medium Risk"
        else:
            risk_label = "🔴 High Risk"

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric("Churn Probability", f"{prob*100:.1f}%")
            st.markdown(f"### {risk_label}")
            if pred == 1:
                st.error("Prediction: Customer is LIKELY TO CHURN")
            else:
                st.success("Prediction: Customer is LIKELY TO STAY")

            st.markdown("##### 💬 Feedback Sentiment (NLP)")
            sentiment_emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}
            st.write(f"{sentiment_emoji.get(sentiment_label,'')} **{sentiment_label}**  (compound score: {scores['compound']:.2f})")

        with res_col2:
            st.markdown("#### 🧠 Why this prediction? (SHAP Explanation)")
            if model_name == "Logistic Regression":
                shap_vals = explainer(X_for_pred)
            else:
                shap_vals = explainer(input_df)

            fig = plt.figure(figsize=(9, 5))
            shap.plots.waterfall(shap_vals[0], max_display=8, show=False)
            fig.tight_layout()
            chart_col, _ = st.columns([3, 1])
            with chart_col:
                st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        # ---- Retention recommendation engine ----
        st.markdown("#### 💡 Suggested Retention Action")
        recommendations = []
        if sentiment_label == "Negative":
            recommendations.append("Flag for immediate human follow-up — feedback sentiment is negative, a strong churn signal.")
        if contract == "Month-to-month":
            recommendations.append("Offer a discount to switch to a 1-year or 2-year contract (long contracts strongly reduce churn).")
        if online_security == "No" and internet_service != "No":
            recommendations.append("Offer a free trial of Online Security add-on.")
        if tech_support == "No" and internet_service != "No":
            recommendations.append("Offer a free Tech Support package for 3 months.")
        if payment_method == "Electronic check":
            recommendations.append("Encourage switching to automatic bank transfer / credit card payment with a small incentive.")
        if monthly_charges > 70:
            recommendations.append("Consider a loyalty discount — customer is on a high monthly plan.")
        if not recommendations:
            recommendations.append("Customer profile looks stable — continue standard engagement.")

        for r in recommendations:
            st.write(f"- {r}")

# =================================================================
# TAB 2 : Dashboard
# =================================================================
with tab2:
    st.subheader("Overall Churn Insights")

    c1, c2, c3, c4 = st.columns(4)
    churn_rate = (raw_df["Churn"] == "Yes").mean() * 100
    c1.metric("Total Customers", len(raw_df))
    c2.metric("Churn Rate", f"{churn_rate:.1f}%")
    c3.metric("Avg Monthly Charges", f"${raw_df['MonthlyCharges'].mean():.2f}")
    c4.metric("Avg Tenure", f"{raw_df['tenure'].mean():.1f} mo")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Churn by Contract Type**")
        ct = pd.crosstab(raw_df["Contract"], raw_df["Churn"], normalize="index") * 100
        st.bar_chart(ct["Yes"])

    with colB:
        st.markdown("**Churn by Internet Service**")
        ct2 = pd.crosstab(raw_df["InternetService"], raw_df["Churn"], normalize="index") * 100
        st.bar_chart(ct2["Yes"])

    st.markdown("**Monthly Charges Distribution by Churn Status**")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    raw_df[raw_df["Churn"] == "No"]["MonthlyCharges"].plot(kind="hist", alpha=0.5, label="Stayed", ax=ax2, bins=30)
    raw_df[raw_df["Churn"] == "Yes"]["MonthlyCharges"].plot(kind="hist", alpha=0.5, label="Churned", ax=ax2, bins=30)
    ax2.legend()
    ax2.set_xlabel("Monthly Charges ($)")
    st.pyplot(fig2)

# =================================================================
# TAB 3 : NLP Feature Impact (the "what's new" proof)
# =================================================================
with tab3:
    st.subheader("🧠 Does the NLP Sentiment Feature Actually Help?")
    st.write(
        "Most churn prediction projects only use structured account data "
        "(contract, charges, tenure, etc). This project adds a **second signal**: "
        "the customer's own words, analyzed with VADER sentiment analysis. "
        "Below is a controlled before/after comparison using the same model "
        "(Logistic Regression) trained on identical data, with and without "
        "the sentiment features."
    )

    impact = load_sentiment_impact()
    comp_df = pd.DataFrame(impact).T
    comp_df.index = ["Without Sentiment (baseline)", "With Sentiment (this project)"]
    st.dataframe(comp_df.style.format("{:.4f}"), use_container_width=True)

    roc_gain = (impact["with_sentiment"]["roc_auc"] - impact["without_sentiment"]["roc_auc"]) * 100
    f1_gain = (impact["with_sentiment"]["f1_score"] - impact["without_sentiment"]["f1_score"]) * 100
    st.success(f"Adding the NLP sentiment feature improved ROC-AUC by **{roc_gain:+.2f} points** "
               f"and F1-Score by **{f1_gain:+.2f} points**.")

    st.markdown("##### Average Sentiment Score: Stayed vs Churned Customers")
    from sentiment import add_sentiment_features
    sent_df = add_sentiment_features(raw_df, text_col="CustomerFeedback")
    avg_sent = sent_df.groupby("Churn")["sentiment_compound"].mean()
    st.bar_chart(avg_sent)
    st.caption("Churned customers' feedback text is, on average, far more negative — confirming sentiment is a genuine churn signal, not just decoration.")