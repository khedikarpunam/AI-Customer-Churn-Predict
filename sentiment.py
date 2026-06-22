"""
sentiment.py
------------
NLP module: extracts sentiment from customer feedback text using VADER
(Valence Aware Dictionary and sEntiment Reasoner).

VADER is a lexicon + rule-based sentiment tool, pre-trained on social-media
style text. No training data is required from us -- it ships ready to use,
which is why it's a great fit for a final-year project (fast, explainable,
no GPU/training needed).

This module is the "additional NLP feature" that sets this project apart
from a typical structured-data-only churn predictor.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def get_sentiment_scores(text):
    """Returns VADER's 4 scores for a piece of text:
    neg, neu, pos (proportions) and compound (overall score from -1 to +1)."""
    if not isinstance(text, str) or text.strip() == "":
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return _analyzer.polarity_scores(text)


def get_sentiment_label(compound_score):
    """Converts the compound score into a simple human-readable label."""
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def add_sentiment_features(df, text_col="CustomerFeedback"):
    """Adds sentiment_compound, sentiment_pos, sentiment_neg, sentiment_neu,
    and sentiment_label columns to a dataframe based on a free-text column."""
    df = df.copy()
    scores = df[text_col].apply(get_sentiment_scores)

    df["sentiment_compound"] = scores.apply(lambda s: s["compound"])
    df["sentiment_pos"] = scores.apply(lambda s: s["pos"])
    df["sentiment_neg"] = scores.apply(lambda s: s["neg"])
    df["sentiment_neu"] = scores.apply(lambda s: s["neu"])
    df["sentiment_label"] = df["sentiment_compound"].apply(get_sentiment_label)

    return df


if __name__ == "__main__":
    # Quick manual test
    samples = [
        "The internet is unreliable and slow, I am switching providers.",
        "I am very happy with the service, great speed and support!",
        "It's okay, nothing special but works fine.",
    ]
    for s in samples:
        scores = get_sentiment_scores(s)
        print(f"{s}\n  -> {scores} -> {get_sentiment_label(scores['compound'])}\n")
