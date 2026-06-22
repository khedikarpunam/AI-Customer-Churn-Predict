"""
preprocess.py
-------------
Loads the Telco Customer Churn + Feedback dataset and cleans/encodes it
so it is ready for ML model training. Also adds NLP sentiment features
derived from each customer's free-text feedback (see sentiment.py).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sentiment import add_sentiment_features


def load_raw_data(path="data/Telco-Customer-Churn-With-Feedback.csv"):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.copy()

    # TotalCharges has some blank strings -> convert to numeric, fill missing with 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop customerID (not a predictive feature)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Drop the prompt column used to originally generate the feedback text
    # (it's metadata about how the text was created, not a real customer feature)
    if "PromptInput" in df.columns:
        df = df.drop(columns=["PromptInput"])

    # Target column: Yes/No -> 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def feature_engineer(df):
    """Adds extra engineered features -> this is what makes the project
    stand out versus a basic structured-data-only churn predictor."""
    df = df.copy()

    # Average monthly spend so far
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # Tenure buckets (new customer / mid / loyal) -> useful for segmentation later
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
    )

    return df


def encode_features(df, encoders=None, fit=True):
    """Label-encodes all categorical (object/category) columns.
    Returns the encoded dataframe and the dict of fitted encoders
    (needed later to encode a single new customer in the web app)."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if "Churn" in cat_cols:
        cat_cols.remove("Churn")

    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    return df, encoders


def get_processed_data(path="data/Telco-Customer-Churn-With-Feedback.csv", use_sentiment=True):
    """Main entry point. Loads data, adds NLP sentiment features (from
    CustomerFeedback text), cleans, engineers features, and encodes.

    Set use_sentiment=False to fall back to the original structured-only
    pipeline (useful for the 'before vs after NLP feature' comparison)."""
    df = load_raw_data(path)

    raw_feedback_text = None
    if use_sentiment and "CustomerFeedback" in df.columns:
        df = add_sentiment_features(df, text_col="CustomerFeedback")
        raw_feedback_text = df["CustomerFeedback"].copy()
        df = df.drop(columns=["CustomerFeedback", "sentiment_label"])
    elif "CustomerFeedback" in df.columns:
        df = df.drop(columns=["CustomerFeedback"])

    df = clean_data(df)
    df = feature_engineer(df)
    df_encoded, encoders = encode_features(df, fit=True)

    if raw_feedback_text is not None:
        df_encoded["_FeedbackTextForDisplay"] = raw_feedback_text.values

    return df_encoded, encoders


if __name__ == "__main__":
    df, encoders = get_processed_data()
    print("Processed data shape:", df.shape)
    print(df.head())
    print("\nChurn distribution:\n", df["Churn"].value_counts(normalize=True))
    if "sentiment_compound" in df.columns:
        print("\nSentiment feature added successfully. Sample stats:")
        print(df["sentiment_compound"].describe())
