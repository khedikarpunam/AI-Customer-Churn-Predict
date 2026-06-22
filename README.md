# 📉 Customer Churn Prediction System (with NLP Sentiment Analysis)
### BTech Final Year Project

An end-to-end Machine Learning system that predicts whether a telecom customer
is likely to **churn** (leave the service) using BOTH structured account data
AND **NLP sentiment analysis of the customer's own feedback text** — then
explains **why**, and recommends a **retention action**, all through an
interactive web dashboard.

---

## 🌟 The Standout Feature: NLP Sentiment Analysis

Most churn projects only look at structured data (contract type, charges,
tenure). This project adds a **second signal**: what the customer actually
*says* — using **VADER sentiment analysis** on customer feedback text.

**Proven impact (not just decoration) — same model, same data, only the
sentiment features were added or removed:**

| Metric | Without Sentiment | With Sentiment | Improvement |
|--------|-------------------|-----------------|-------------|
| Accuracy | 79.4% | **89.6%** | +10.2 pts |
| F1-Score | 58.5% | **78.5%** | **+20.0 pts** |
| ROC-AUC | 84.6% | **93.0%** | **+8.4 pts** |

This is the single most important chart for your viva: it **proves** the NLP
addition genuinely improves predictions instead of just being a feature for
the sake of it.

---

## 🌟 All Additional Features (vs a typical/basic churn project)

| # | Feature | What it does |
|---|---------|---------------|
| 1 | **NLP Sentiment Analysis (VADER)** | Extracts sentiment from customer feedback text and feeds it into the model |
| 2 | **Multi-model comparison** | Trains Logistic Regression, Random Forest, and XGBoost; auto-selects the best by ROC-AUC |
| 3 | **Explainable AI (SHAP)** | Shows *exactly which factors* pushed a specific customer toward churn |
| 4 | **Risk Segmentation** | Classifies every customer as Low / Medium / High churn risk |
| 5 | **Retention Recommendation Engine** | Rule-based suggestions generated from churn drivers + sentiment |
| 6 | **Engineered Features** | `AvgMonthlySpend`, `TenureGroup` — extra signals beyond raw columns |
| 7 | **Interactive Dashboard** | Live charts on churn-by-contract, churn-by-internet-service, sentiment-by-churn |
| 8 | **Before/After NLP Comparison Tab** | Live proof inside the app that sentiment improves the model |

---

## 🗂️ Project Structure

```
churn_project/
├── data/
│   ├── Telco-Customer-Churn.csv                  # Original structured-only dataset
│   └── Telco-Customer-Churn-With-Feedback.csv    # Same data + CustomerFeedback text column
├── src/
│   ├── preprocess.py     # Cleaning, feature engineering, encoding, sentiment integration
│   ├── sentiment.py       # NLP module: VADER sentiment scoring
│   └── train_model.py     # Trains & compares 3 models + before/after sentiment comparison
├── models/                # Auto-generated after training (pkl files)
├── outputs/
│   ├── model_comparison.json     # Metrics for all 3 models
│   └── sentiment_impact.json     # Before/after sentiment comparison numbers
├── app.py                 # Streamlit web application (3 tabs)
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run (Step by Step)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
cd churn_project
python src/train_model.py
```
This will:
- Load and clean the dataset (with NLP sentiment features)
- Train Logistic Regression, Random Forest, and XGBoost
- Print a comparison table (accuracy, precision, recall, F1, ROC-AUC)
- Run the before/after sentiment comparison and print the improvement
- Save the best model + SHAP explainer into `models/`

### 3. Launch the web app
```bash
streamlit run app.py
```
Open the local URL shown in the terminal (usually `http://localhost:8501`).
Go to the **Predict Churn** tab, fill in a customer's details, type some
feedback text (try writing something negative vs positive and compare!),
and hit Predict.

---

## 📊 Model Results (with NLP sentiment feature, on this dataset)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** ⭐ | 91.3% | 87.7% | 78.1% | 82.6% | **95.6%** |
| Random Forest | 90.9% | 91.8% | 72.2% | 80.8% | 94.4% |
| Logistic Regression | 89.6% | 86.7% | 71.7% | 78.5% | 93.0% |

> XGBoost was auto-selected as the best model based on ROC-AUC.
> Note how much higher these numbers are versus a structured-data-only
> model (~84.6% ROC-AUC) — this is the NLP feature paying off.

---

## 🧠 How the NLP Sentiment Feature Works

1. Each customer has a `CustomerFeedback` text field (a review/feedback/complaint).
2. **VADER** (`vaderSentiment` library) scores the text on 4 dimensions:
   `neg`, `neu`, `pos` (proportions) and `compound` (overall sentiment from -1 to +1).
3. These 4 scores are added as **new numeric features** to the model, alongside
   the structured account data (contract, tenure, charges, etc.).
4. The model learns that negative-sentiment feedback strongly correlates with churn
   (in this dataset: churned customers average a sentiment score of **0.14**,
   while retained customers average **0.72** — a huge gap).

**Why VADER and not a deep learning model?** VADER is lexicon + rule-based,
needs no training data of our own, runs instantly, and is easy to explain in
a viva — a deliberate, defensible choice for a final-year project, not a
limitation.

---

## 🧠 How the Explainability (SHAP) Works

We use **SHAP (SHapley Additive exPlanations)**, a game-theory-based method
that shows how much each feature pushed a prediction up or down for
*one specific customer* — not just globally important features overall.
This is what the waterfall chart in the app visualizes, and now includes
the sentiment features as explainable inputs too.

---

## 🔁 How the Retention Recommendation Engine Works

A rule-based system layered on top of the model output:
- If feedback sentiment is **Negative** → flag for immediate human follow-up
- If `Contract == Month-to-month` → suggest contract upgrade offer
- If `OnlineSecurity == No` → suggest free security add-on trial
- If `PaymentMethod == Electronic check` → suggest auto-pay incentive
- If `MonthlyCharges` is high → suggest loyalty discount

---

## 📚 Dataset Sources

1. **IBM Telco Customer Churn dataset** — 7,043 customers, structured account/demographic data.
2. **Telco Customer Churn + Realistic Customer Feedback** (Kaggle, by beatafaron) —
   the same 7,043 customers enriched with AI-generated, realistic `CustomerFeedback`
   text per customer, used as the NLP input for sentiment analysis.

---

## 📈 Possible Future Enhancements (good "future scope" talking points)

- Replace VADER with a fine-tuned BERT model for even better sentiment accuracy
- Deploy the app on Streamlit Cloud / Render for a live public link
- Add a database to store predictions and feedback over time
- Add topic modeling (LDA) to auto-categorize *why* customers are unhappy
  (e.g., billing, speed, support) — not just positive/negative
- Hyperparameter tuning with GridSearchCV / Optuna

---

## 🎓 Suggested Viva / Presentation Talking Points

1. **Problem statement**: Why churn prediction matters (cost of acquiring a new customer vs retaining one)
2. **The NLP addition**: Most projects use only structured data; this one adds sentiment from feedback text — and *proves* it helps with a controlled before/after comparison (+8.4 ROC-AUC points)
3. **Why VADER**: No training data needed, fast, explainable, well-suited to short feedback/review text — a deliberate engineering choice
4. **Why these 3 algorithms**: Logistic Regression (interpretable baseline), Random Forest (handles non-linearity), XGBoost (industry-standard for tabular data, won here)
5. **Why ROC-AUC over accuracy**: Dataset is imbalanced (~26.5% churn), so accuracy alone is misleading
6. **What SHAP adds**: Moves the project from "black box prediction" to "explainable AI"
7. **Business value**: Retention recommendations + sentiment flagging turn the model into a decision-support tool a real call-center team could use
