"""
train.py — Full training pipeline for Customer Churn Prediction
Run: python train.py
Outputs: xgboost_model.pkl, scaler.pkl, label_encoders.pkl,
         shap_importance.csv, predictions.csv, model_results.json
"""
import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

print("=" * 50)
print("Customer Churn Prediction — Training Pipeline")
print("=" * 50)

# ── 1. LOAD DATA ─────────────────────────────────────
df = pd.read_csv('ecommerce_data.csv', parse_dates=['invoice_date'])
print(f"\n[1/7] Data loaded: {df.shape[0]:,} transactions, {df['customer_id'].nunique():,} customers")

# ── 2. RFM FEATURE ENGINEERING ───────────────────────
snapshot    = pd.Timestamp('2024-12-31')
CHURN_DAYS  = 90

rfm = df.groupby('customer_id').agg(
    last_purchase          = ('invoice_date', 'max'),
    frequency              = ('invoice_no',   'nunique'),
    monetary               = ('total_price',  'sum'),
    avg_order_value        = ('total_price',  'mean'),
    total_quantity         = ('quantity',     'sum'),
    first_purchase         = ('invoice_date', 'min'),
    n_categories           = ('category',     'nunique'),
    country                = ('country',      lambda x: x.mode()[0]),
    customer_segment       = ('customer_segment', lambda x: x.mode()[0]),
).reset_index()

rfm['recency']                = (snapshot - rfm['last_purchase']).dt.days
rfm['customer_lifespan_days'] = (rfm['last_purchase'] - rfm['first_purchase']).dt.days
rfm['purchase_frequency_rate']= rfm['frequency'] / (rfm['customer_lifespan_days'].replace(0,1) / 30)
rfm['log_monetary']           = np.log1p(rfm['monetary'])
rfm['log_frequency']          = np.log1p(rfm['frequency'])
rfm['log_avg_order']          = np.log1p(rfm['avg_order_value'])
rfm['churned']                = (rfm['recency'] > CHURN_DAYS).astype(int)

print(f"[2/7] RFM features engineered: {len(rfm):,} customers, churn rate = {rfm['churned'].mean():.1%}")

# ── 3. ENCODE CATEGORICALS ────────────────────────────
le_country = LabelEncoder()
le_segment = LabelEncoder()
rfm['country_enc'] = le_country.fit_transform(rfm['country'])
rfm['segment_enc'] = le_segment.fit_transform(rfm['customer_segment'])

FEATURE_COLS = [
    'recency', 'log_frequency', 'log_monetary', 'log_avg_order',
    'total_quantity', 'n_categories', 'customer_lifespan_days',
    'purchase_frequency_rate', 'country_enc', 'segment_enc'
]
X = rfm[FEATURE_COLS].fillna(0)
y = rfm['churned']

# ── 4. TRAIN / TEST SPLIT + SMOTE ─────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# SMOTE on training set ONLY — never apply to test data
smote       = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train_sm)
X_test_sc   = scaler.transform(X_test)
print(f"[3/7] Train/test split done. After SMOTE: {X_train_sm.shape[0]:,} train samples")

# ── 5. TRAIN 3 MODELS ────────────────────────────────
print("[4/7] Training models...")
models_cfg = {
    'Logistic Regression': (LogisticRegression(C=0.1, random_state=42, max_iter=1000), True),
    'Random Forest':       (RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42), False),
    'XGBoost':             (XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                          subsample=0.8, colsample_bytree=0.8,
                                          random_state=42, eval_metric='logloss', verbosity=0), False),
}
results = {}
for name, (model, use_scaler) in models_cfg.items():
    Xtr = X_train_sc if use_scaler else X_train_sm
    Xte = X_test_sc  if use_scaler else X_test
    model.fit(Xtr, y_train_sm)
    proba = model.predict_proba(Xte)[:, 1]
    preds = (proba >= 0.5).astype(int)
    rep   = classification_report(y_test, preds, output_dict=True)
    results[name] = {
        'model': model,
        'proba': proba,
        'auc':   round(roc_auc_score(y_test, proba), 4),
        'ap':    round(average_precision_score(y_test, proba), 4),
        'f1':    round(rep['1']['f1-score'], 4),
        'precision': round(rep['1']['precision'], 4),
        'recall':    round(rep['1']['recall'], 4),
    }
    print(f"  {name:22s}  AUC={results[name]['auc']:.4f}  F1={results[name]['f1']:.4f}")

# ── 6. SHAP VALUES ────────────────────────────────────
print("[5/7] Computing SHAP values...")
xgb_model   = results['XGBoost']['model']
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap_imp    = pd.DataFrame({
    'feature':       FEATURE_COLS,
    'mean_abs_shap': np.abs(shap_values).mean(0)
}).sort_values('mean_abs_shap', ascending=False)

# ── 7. REVENUE AT RISK ────────────────────────────────
test_df = rfm.iloc[X_test.index].copy()
test_df['churn_probability'] = results['XGBoost']['proba']
test_df['risk_tier']         = pd.cut(test_df['churn_probability'],
                                       bins=[0, 0.3, 0.6, 1.0],
                                       labels=['Low','Medium','High'])
test_df['revenue_at_risk_score'] = (
    test_df['churn_probability'] *
    test_df['avg_order_value']   *
    test_df['purchase_frequency_rate']
)
high_risk_rev = test_df[test_df['risk_tier']=='High']['revenue_at_risk_score'].sum()
print(f"[6/7] Revenue at Risk (High tier): Rp {high_risk_rev:,.0f}")

# ── 8. SAVE ARTEFACTS ─────────────────────────────────
pickle.dump(xgb_model,                              open('xgboost_model.pkl','wb'))
pickle.dump(scaler,                                 open('scaler.pkl','wb'))
pickle.dump({'country': le_country, 'segment': le_segment}, open('label_encoders.pkl','wb'))
shap_imp.to_csv('shap_importance.csv', index=False)
test_df.to_csv('predictions.csv', index=False)
rfm.to_csv('rfm_features.csv', index=False)

model_results_out = {n: {k: v for k, v in results[n].items() if k not in ('model','proba')}
                     for n in results}
with open('model_results.json','w') as f:
    json.dump(model_results_out, f, indent=2)

print("[7/7] All artefacts saved.")
print("\nDone. Run: streamlit run app.py")
print("=" * 50)
