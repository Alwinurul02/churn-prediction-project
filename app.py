import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Churn & Revenue Risk Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── IBCS matplotlib defaults ─────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor':   'none',
    'axes.edgecolor':   '#cccccc',
    'axes.linewidth':   0.5,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'DejaVu Sans',
    'font.size':        9,
    'xtick.color':      '#666666',
    'ytick.color':      '#666666',
    'text.color':       '#1a1a1a',
})

IBCS_FILLED  = '#1a1a1a'
IBCS_HOLLOW  = 'white'
IBCS_EDGE    = '#1a1a1a'
IBCS_MUTED   = '#888888'
IBCS_MONO    = 'DejaVu Sans Mono'

# ── GLOBAL CSS ───────────────────────────────────────
st.markdown("""
<style>
  section[data-testid="stSidebar"] { background: #f9f9f9; }
  .block-container { padding-top: 1.5rem; }
  .stMetric { background: #f5f5f5; border-radius: 8px; padding: 12px 16px; }
  div[data-testid="stMetricValue"] > div { font-family: 'DejaVu Sans Mono', monospace; }
  .ibcs-note { font-size: 11px; color: #999; font-family: monospace; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── LOAD ARTEFACTS ───────────────────────────────────
@st.cache_resource
def load_artefacts():
    model    = pickle.load(open('xgboost_model.pkl', 'rb'))
    scaler   = pickle.load(open('scaler.pkl', 'rb'))
    encoders = pickle.load(open('label_encoders.pkl', 'rb'))
    return model, scaler, encoders

@st.cache_data
def load_data():
    rfm    = pd.read_csv('rfm_features.csv')
    preds  = pd.read_csv('predictions.csv')
    shap_i = pd.read_csv('shap_importance.csv')
    with open('model_results.json') as f:
        results = json.load(f)
    return rfm, preds, shap_i, results

try:
    model, scaler, encoders = load_artefacts()
    rfm, preds, shap_imp, results = load_data()
except Exception as e:
    st.error(f"Model files not found. Run `python train.py` first.\n\n{e}")
    st.stop()

FEATURE_COLS = [
    'recency', 'log_frequency', 'log_monetary', 'log_avg_order',
    'total_quantity', 'n_categories', 'customer_lifespan_days',
    'purchase_frequency_rate', 'country_enc', 'segment_enc'
]

COUNTRIES = list(encoders['country'].classes_)
SEGMENTS  = list(encoders['segment'].classes_)

# ── IBCS CHART HELPERS ───────────────────────────────
def ibcs_hbar(ax, labels, values, benchmark=None, title='', footnote='', xlim_pad=1.25):
    avg = benchmark if benchmark is not None else np.mean(values)
    colors = [IBCS_FILLED if v >= avg else IBCS_HOLLOW for v in values]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   edgecolor=IBCS_EDGE, height=0.5, linewidth=1.2)
    if benchmark is not None:
        ax.axvline(benchmark, color=IBCS_EDGE, linestyle='--', linewidth=0.8, alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(values)*0.02, bar.get_y() + bar.get_height()/2,
                f'{w:.1f}%', va='center', fontsize=8, fontfamily=IBCS_MONO, fontweight='bold')
    ax.set_xlim(0, max(values) * xlim_pad)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
    if footnote:
        ax.text(0.99, -0.16, footnote, transform=ax.transAxes,
                ha='right', fontsize=7, color='#aaa')
    ax.spines['left'].set_color('#ccc')
    ax.spines['bottom'].set_color('#ccc')

def ibcs_shap_bar(ax, features, values, title=''):
    bars = ax.barh(features[::-1], values[::-1], color=IBCS_FILLED,
                   edgecolor=IBCS_EDGE, height=0.5, linewidth=0.8)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(values)*0.02, bar.get_y() + bar.get_height()/2,
                f'{w:.2f}', va='center', fontsize=8, fontfamily=IBCS_MONO, color=IBCS_MUTED)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
    ax.spines['left'].set_color('#ccc')
    ax.spines['bottom'].set_color('#ccc')

# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛒 Churn Predictor")
    st.caption("E-Commerce · SEA Market · n=4,000")
    page = st.radio("Navigation", [
        "Overview",
        "Predict customer",
        "Model performance",
        "Revenue at risk"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Model: XGBoost · AUC {results['XGBoost']['auc']:.4f}")
    st.caption(f"F1: {results['XGBoost']['f1']:.4f} · Snapshot: 2024-12-31")

# ════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("## Customer Churn & Revenue Risk Report")
    st.caption("E-Commerce · SEA Market · Snapshot 2024-12-31 · n=4,000 customers")
    st.divider()

    total   = len(rfm)
    churned = int(rfm['churned'].sum())
    rate    = rfm['churned'].mean() * 100
    high_n  = int((preds['risk_tier'] == 'High').sum())
    rev_risk= preds[preds['risk_tier'] == 'High']['revenue_at_risk_score'].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers",      f"{total:,}")
    c2.metric("Churn rate",           f"{rate:.1f}%",   f"{churned:,} churned")
    c3.metric("High-risk customers",  f"{high_n:,}",    "in test cohort (n=800)")
    c4.metric("Revenue at risk (Rp)", f"{rev_risk/1e6:.1f}M", "high-risk tier")

    st.info(
        "**Recency** dominates churn prediction with SHAP weight **7.12**. 12× higher than the "
        "next feature. Customers inactive ≥91 days churn with 100% precision. "
        "Budget segment churns at **39.2%** vs Regular **21.7%**, a **1.8×** differential "
        "demanding separate retention strategies."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        seg = rfm.groupby('customer_segment')['churned'].mean().sort_values(ascending=False) * 100
        ibcs_hbar(ax, list(seg.index), list(seg.values),
                  benchmark=rate,
                  title='Budget customers churn 1.8× faster than Regular',
                  footnote='AC 2024 · filled = above avg (27.0%) · hollow = below avg')
        ax.set_xlabel('Churn rate (%)', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
        countries = rfm.groupby('country')['churned'].mean().sort_values(ascending=False) * 100
        ibcs_hbar(ax2, list(countries.index), list(countries.values),
                  benchmark=rate,
                  title='Indonesia and Malaysia have highest churn rates',
                  footnote='AC 2024 · filled = above avg · hollow = below avg')
        ax2.set_xlabel('Churn rate (%)', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        fig3, ax3 = plt.subplots(figsize=(5.5, 3.2))
        active_rev  = rfm[rfm['churned']==0]['log_monetary']
        churned_rev = rfm[rfm['churned']==1]['log_monetary']
        ax3.hist(active_rev,  bins=30, alpha=0.85, color=IBCS_FILLED, density=True, label='Active')
        ax3.hist(churned_rev, bins=30, alpha=0.45, color=IBCS_MUTED,  density=True, label='Churned')
        ax3.set_title('Churned customers concentrate in lower revenue bands', fontsize=9, fontweight='bold')
        ax3.set_xlabel('Log revenue (Rp)')
        ax3.set_ylabel('Density')
        ax3.legend(frameon=False, fontsize=8)
        ax3.spines['left'].set_color('#ccc')
        ax3.spines['bottom'].set_color('#ccc')
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(5.5, 3.2))
        rfm2 = rfm.copy()
        rfm2['rb'] = pd.cut(rfm2['recency'], bins=[0,30,60,90,180,365,9999],
                            labels=['0–30','31–60','61–90','91–180','181–365','365+'])
        rec = rfm2.groupby('rb', observed=True)['churned'].mean() * 100
        colors = [IBCS_FILLED if v > 50 else IBCS_HOLLOW for v in rec.values]
        bars = ax4.bar(rec.index, rec.values, color=colors,
                       edgecolor=IBCS_EDGE, width=0.6, linewidth=1.2)
        for bar in bars:
            ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                     f'{bar.get_height():.0f}%', ha='center', fontsize=8,
                     fontfamily=IBCS_MONO, fontweight='bold')
        ax4.set_title('Customers inactive >90 days churn at 100%', fontsize=9, fontweight='bold')
        ax4.set_xlabel('Days since last purchase')
        ax4.set_ylabel('Churn rate (%)')
        ax4.set_ylim(0, 120)
        ax4.spines['left'].set_color('#ccc')
        ax4.spines['bottom'].set_color('#ccc')
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)

    st.markdown('<p class="ibcs-note">AC = Actual · Filled bars = above benchmark · Hollow bars = below benchmark · IBCS notation</p>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# PAGE 2 — PREDICT CUSTOMER
# ════════════════════════════════════════════════════
elif page == "Predict customer":
    st.markdown("## Predict churn risk for a customer")
    st.caption("Enter customer details to get churn probability and revenue at risk.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Purchase history")
        recency   = st.slider("Days since last purchase (recency)", 0, 365, 45)
        frequency = st.slider("Total number of orders (frequency)", 1, 50, 12)
        monetary  = st.number_input("Total revenue spent (Rp)", 10_000, 100_000_000, 4_500_000, step=100_000, format="%d")
        avg_order = st.number_input("Average order value (Rp)", 10_000, 20_000_000, 380_000, step=10_000, format="%d")
    with col2:
        st.markdown("#### Customer profile")
        total_qty = st.slider("Total items purchased", 1, 200, 30)
        n_cats    = st.slider("Number of product categories purchased from", 1, 8, 3)
        lifespan  = st.slider("Customer lifespan (days)", 0, 730, 200)
        country   = st.selectbox("Country", COUNTRIES)
        segment   = st.selectbox("Customer segment", SEGMENTS)

    if st.button("Predict churn risk", type="primary", use_container_width=True):
        freq_rate    = frequency / max(lifespan / 30, 1)
        country_enc  = int(encoders['country'].transform([country])[0])
        segment_enc  = int(encoders['segment'].transform([segment])[0])

        # Build payload with ALL 10 features — matching training exactly
        features = pd.DataFrame([[
            recency,
            np.log1p(frequency),
            np.log1p(monetary),
            np.log1p(avg_order),
            total_qty,
            n_cats,
            lifespan,
            freq_rate,
            country_enc,
            segment_enc
        ]], columns=FEATURE_COLS)

        churn_prob  = float(model.predict_proba(features)[0][1])
        risk_tier   = 'High' if churn_prob > 0.6 else 'Medium' if churn_prob > 0.3 else 'Low'
        rev_at_risk = churn_prob * avg_order * freq_rate

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn probability", f"{churn_prob*100:.1f}%")
        r2.metric("Risk tier", risk_tier)
        r3.metric("Revenue at risk", f"Rp {rev_at_risk:,.0f}")

        # Visual gauge
        tier_color = {'High':'#c0392b','Medium':'#d68910','Low':'#1e8449'}[risk_tier]
        st.markdown(f"""
        <div style="background:#f5f5f5;border-radius:8px;padding:14px;margin-top:8px">
          <p style="font-size:11px;color:#888;margin-bottom:6px;font-family:monospace">Churn probability</p>
          <div style="background:#e0e0e0;border-radius:4px;height:12px;overflow:hidden">
            <div style="background:{tier_color};width:{churn_prob*100:.0f}%;height:100%;border-radius:4px"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:5px">
            <span style="font-size:10px;color:#aaa;font-family:monospace">0% — Low</span>
            <span style="font-size:12px;font-weight:500;color:{tier_color};font-family:monospace">{churn_prob*100:.1f}%</span>
            <span style="font-size:10px;color:#aaa;font-family:monospace">High — 100%</span>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Top churn drivers (SHAP)")
        feature_names_display = {
            'recency':'Recency (days inactive)',
            'customer_lifespan_days':'Customer lifespan',
            'segment_enc':'Customer segment',
            'n_categories':'Category diversity',
            'purchase_frequency_rate':'Purchase frequency',
            'log_monetary':'Total revenue (log)',
            'log_frequency':'Order count (log)',
            'log_avg_order':'Avg order value (log)',
        }
        for feat in shap_imp.head(5)['feature']:
            label = feature_names_display.get(feat, feat)
            imp   = float(shap_imp[shap_imp['feature']==feat]['mean_abs_shap'].values[0])
            icon  = "🔴" if feat=='recency' and recency > 90 else "🟡" if feat=='recency' and recency > 45 else "🟢"
            st.markdown(f"{icon} **{label}** — global importance {imp:.3f}")

        st.markdown("---")
        st.markdown("#### Recommended action")
        if risk_tier == 'High':
            st.error("Immediate action required")
            st.markdown("- Send personalised win-back email with 15–20% discount\n- Trigger push notification with exclusive offer\n- Assign to CRM for personal outreach if high value")
        elif risk_tier == 'Medium':
            st.warning("Monitor and re-engage")
            st.markdown("- Send product recommendation email\n- Offer loyalty points bonus on next purchase\n- Enrol in re-engagement campaign sequence")
        else:
            st.success("Low risk — focus on upsell")
            st.markdown("- Promote to loyalty / premium tier\n- Show cross-category recommendations\n- Request review or referral")

# ════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════
elif page == "Model performance":
    st.markdown("## Model performance")
    st.caption("Comparison of all 3 trained models — AUC, F1, Precision, Recall")
    st.divider()

    # Metrics table
    df_res = pd.DataFrame({
        'Model':             list(results.keys()),
        'AUC-ROC':           [results[m]['auc'] for m in results],
        'Avg precision':     [results[m]['ap'] for m in results],
        'Precision (churn)': [results[m]['precision'] for m in results],
        'Recall (churn)':    [results[m]['recall'] for m in results],
        'F1 (churn)':        [results[m]['f1'] for m in results],
    })
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### AUC-ROC & F1 by model")
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        model_names = ['XGBoost', 'Random Forest', 'Logistic Regression']
        auc_vals = [results[m]['auc'] for m in model_names]
        f1_vals  = [results[m]['f1'] for m in model_names]
        y_pos    = np.arange(len(model_names))
        ax.barh(y_pos - 0.18, auc_vals, height=0.3, color=IBCS_FILLED,
                edgecolor=IBCS_EDGE, label='AUC-ROC')
        ax.barh(y_pos + 0.18, f1_vals,  height=0.3, color=IBCS_HOLLOW,
                edgecolor=IBCS_EDGE, linewidth=1.2, label='F1-score (churn)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=9)
        ax.set_xlim(0.88, 1.06)
        ax.set_title('XGBoost selected — AUC 1.000 + SHAP support', fontsize=9, fontweight='bold')
        for i, (a, f) in enumerate(zip(auc_vals, f1_vals)):
            ax.text(a+0.003, i-0.18, f'{a:.4f}', va='center', fontsize=7, fontfamily=IBCS_MONO)
            ax.text(f+0.003, i+0.18, f'{f:.4f}', va='center', fontsize=7, fontfamily=IBCS_MONO, color=IBCS_MUTED)
        ax.legend(frameon=False, fontsize=8)
        ax.invert_yaxis()
        ax.spines['left'].set_color('#ccc')
        ax.spines['bottom'].set_color('#ccc')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("#### SHAP feature importance")
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
        feature_labels = {
            'recency':'Recency (days)',
            'customer_lifespan_days':'Customer lifespan',
            'segment_enc':'Customer segment',
            'n_categories':'Category diversity',
            'purchase_frequency_rate':'Purchase frequency',
            'log_monetary':'Total revenue (log)',
            'log_frequency':'Order count (log)',
            'log_avg_order':'Avg order value (log)',
            'country_enc':'Country',
            'total_quantity':'Total items',
        }
        top8 = shap_imp.head(8).copy()
        top8['label'] = top8['feature'].map(feature_labels).fillna(top8['feature'])
        ibcs_shap_bar(ax2, list(top8['label']), list(top8['mean_abs_shap']),
                      title='Recency dominates — 7.12 vs next feature 0.59')
        ax2.set_xlabel('Mean |SHAP value|', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Methodology notes")
    st.markdown("""
    - **SMOTE** applied to training set only — prevents test contamination
    - **Stratified K-Fold** used for robust train/test split
    - **SHAP TreeExplainer** used for per-customer attribution
    - **10 features** engineered from raw transactions (RFM + lifespan + category diversity)
    - **Churn label**: recency > 90 days = churned (operational threshold)
    """)

    st.markdown('<p class="ibcs-note">Filled = selected model · hollow = alternative · SMOTE on train only · 90-day churn window</p>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# PAGE 4 — REVENUE AT RISK
# ════════════════════════════════════════════════════
elif page == "Revenue at risk":
    st.markdown("## Revenue at risk analysis")
    st.caption("Quantifying the business impact of predicted churn · test cohort n=800")
    st.divider()

    high_df = preds[preds['risk_tier'] == 'High']
    low_df  = preds[preds['risk_tier'] == 'Low']

    r1, r2, r3 = st.columns(3)
    r1.metric("High-risk revenue",   f"Rp {high_df['revenue_at_risk_score'].sum()/1e6:.1f}M", f"{len(high_df)} customers")
    r2.metric("Low-risk revenue",    f"Rp {low_df['revenue_at_risk_score'].sum()/1e6:.1f}M",  f"{len(low_df)} customers")
    r3.metric("Avg revenue per high-risk account", f"Rp {high_df['revenue_at_risk_score'].mean():,.0f}")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### Top 20 priority retention list")
        top20 = preds.nlargest(20, 'revenue_at_risk_score')[[
            'customer_id','churn_probability','risk_tier','recency','frequency','monetary','revenue_at_risk_score'
        ]].copy()
        top20['churn_probability']    = top20['churn_probability'].map('{:.1%}'.format)
        top20['monetary']             = top20['monetary'].map('Rp {:,.0f}'.format)
        top20['revenue_at_risk_score']= top20['revenue_at_risk_score'].map('Rp {:,.0f}'.format)
        top20.columns = ['Customer ID','Churn prob','Risk tier','Recency (days)','Orders','Total revenue','Revenue at risk']
        st.dataframe(top20, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Revenue by risk tier (Rp M)")
        fig, ax = plt.subplots(figsize=(4, 2.0))
        tiers = ['High', 'Low']
        revs  = [preds[preds['risk_tier']==t]['revenue_at_risk_score'].sum()/1e6 for t in tiers]
        colors= [IBCS_FILLED, IBCS_HOLLOW]
        bars  = ax.barh(tiers[::-1], revs[::-1], color=colors[::-1],
                        edgecolor=IBCS_EDGE, height=0.45, linewidth=1.2)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + max(revs)*0.02, bar.get_y() + bar.get_height()/2,
                    f'Rp {w:.1f}M', va='center', fontsize=8, fontfamily=IBCS_MONO, fontweight='bold')
        ax.set_xlim(0, max(revs) * 1.3)
        ax.set_title('Rp 143.6M concentrated in high-risk tier', fontsize=9, fontweight='bold')
        ax.spines['left'].set_color('#ccc')
        ax.spines['bottom'].set_color('#ccc')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Business recommendation")
        st.markdown("""
        Recovering **30%** of high-risk revenue through targeted win-back campaigns = **Rp 43M** recovered vs Rp 60M+ cost of blanket discounts.

        **Retention strategy by tier:**
        - 🔴 **High** — immediate outreach + 15–20% discount
        - 🟡 **Medium** — automated email sequence
        - 🟢 **Low** — loyalty upsell campaign
        """)

    st.markdown("""
    <p class="ibcs-note">
    Revenue at risk = P(churn) × avg_order_value × monthly_frequency<br>
    AC = Actual · Filled = primary measure · Hollow = reference measure
    </p>""", unsafe_allow_html=True)
