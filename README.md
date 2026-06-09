# 🛒 Customer Churn & Revenue Risk Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![AUC](https://img.shields.io/badge/AUC--ROC-1.000-brightgreen?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**Live demo → [churn-prediction-project.streamlit.app](https://churn-prediction-project-wtvyys5jhgmeqpdwzr4aqg.streamlit.app)**

End to end ML system that predicts which e-commerce customers are likely to churn in the next 90 days and quantifies the revenue at risk, enabling data driven retention campaigns.

---

## I. Strategic business insights

By mapping ML inferences directly to financial metrics, the system uncovers high leverage operational insights:

- **Segment differential:** Budget customers churn at **39.2%** vs Regular **21.7%** (1.8× gap). Price led win back campaigns should target the Budget tier only not the entire base.
- **Revenue at risk:** The model isolates a high risk cohort representing **Rp 143.6M** in absolute revenue at risk, giving the finance team an exact budget justification for retention spend.
- **Revenue gap:** Active customers average **Rp 9.80M** revenue vs **Rp 7.03M** for churned profiles, a **−Rp 2.77M (−28.3%)** variance per lost account, proving retention ROI exceeds acquisition cost.
- **Recency threshold:** Customers inactive for **>90 days** churn with 100% precision. This is the operational trigger date for all retention actions.

---

## II. Methodology

```
Raw transactions (50K)
    ↓ Data cleaning
RFM feature engineering (10 features)
    ↓ Churn labelling (90-day rule)
Train/test split (80/20, stratified)
    ↓ SMOTE — training set only
3 models trained and compared
    ↓ XGBoost selected (AUC 1.000)
SHAP explainability
    ↓
Revenue at Risk = P(churn) × avg_order × monthly_frequency
    ↓
Streamlit app + Streamlit Cloud deployment
```

### Features engineered

| Feature | Description |
|---|---|
| `recency` | Days since last purchase |
| `log_frequency` | Log of unique order count |
| `log_monetary` | Log of total spend |
| `log_avg_order` | Log of mean order value |
| `total_quantity` | Total items purchased |
| `n_categories` | Category diversity (loyalty proxy) |
| `customer_lifespan_days` | First to last purchase span |
| `purchase_frequency_rate` | Orders per month |
| `country_enc` | Geographic segment (label encoded) |
| `segment_enc` | Premium / Regular / Budget (label encoded) |

### Model comparison

| Model | AUC-ROC | F1 (churn) | Notes |
|---|---|---|---|
| Logistic Regression | 0.998 | 0.951 | Baseline |
| Random Forest | 1.000 | 1.000 | Strong but less explainable |
| **XGBoost ✓** | **1.000** | **1.000** | Selected — SHAP compatible |

> **Why XGBoost over Random Forest?** Both achieve AUC 1.000. XGBoost is selected because SHAP's TreeExplainer provides per-customer attribution, critical for business stakeholders who need to act on individual predictions, not just model averages.

---

## III. Dashboard (IBCS style)

The app follows [IBCS](https://www.ibcs.com) (International Business Communication Standards):
- Filled bars = values above benchmark / primary measure
- Hollow bars = values below benchmark / reference measure  
- No decorative colour — all meaning encoded in shape and position
- Insight-first chart titles ("Budget customers churn 1.8× faster", not "Churn Rate by Segment")

**4 pages:**
1. **Overview** — KPI matrix, churn by segment, country, revenue distribution, recency analysis
2. **Predict customer** — Interactive risk engine with all 10 features, SHAP explanation, action recommendation
3. **Model performance** — AUC/F1 comparison, SHAP feature importance, methodology notes
4. **Revenue at risk** — Priority retention table, revenue by tier, business recommendation

---

## IV. Quick start

```bash
# 1. Clone
git clone https://github.com/Alwinurul02/churn-prediction-project.git
cd churn-prediction-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain model from scratch
python train.py

# 4. Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## V. Repository structure

```
churn-prediction-project/
├── notebooks/
│   ├── 01_eda.ipynb          ← Data cleaning, RFM, IBCS EDA charts
│   └── 02_modeling.ipynb     ← SMOTE, model training, SHAP, evaluation
├── data/
│   └── README.md             ← Dataset source and download instructions
├── app.py                    ← Streamlit application (4 pages)
├── train.py                  ← Reproducible training pipeline
├── ecommerce_data.csv        ← Synthetic dataset (50K transactions)
├── rfm_features.csv          ← Engineered RFM features
├── predictions.csv           ← Test set predictions + risk tiers
├── xgboost_model.pkl         ← Trained XGBoost classifier
├── scaler.pkl                ← StandardScaler (for Logistic Regression)
├── label_encoders.pkl        ← LabelEncoders for country and segment
├── shap_importance.csv       ← SHAP feature importance scores
├── model_results.json        ← Evaluation metrics for all 3 models
├── requirements.txt
└── README.md
```

---

## VI. Tech stack

| Layer | Tool |
|---|---|
| Data processing | pandas, numpy |
| ML modelling | scikit-learn, XGBoost |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualisation | matplotlib (IBCS style) |
| App framework | Streamlit |
| Deployment | Streamlit Cloud |

---

*Dataset: Synthetic e-commerce transaction data (50,000 rows, 4,000 customers) modelled after Southeast Asian market patterns.*
