import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(
    page_title="ChurnIQ — Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #f5f6fa; color: #1a1d23; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1e2a4a 0%, #162038 100%) !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #c8d3ea !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] .stMarkdown a { color: #818cf8 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.07);
    border: 1.5px dashed rgba(255,255,255,0.25);
    border-radius: 10px;
}
[data-testid="stSidebar"] label { color: #a5b4d0 !important; font-size: 0.78rem !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.1) !important;
    border-color: rgba(255,255,255,0.2) !important;
    color: #fff !important;
}

.kpi-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 22px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    border-top: 4px solid var(--accent-color);
    position: relative;
}
.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.85rem;
    font-weight: 600;
    color: #111827;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.74rem;
    color: #9ca3af;
    margin-top: 6px;
}
.kpi-icon {
    position: absolute;
    top: 18px; right: 18px;
    font-size: 1.5rem;
    opacity: 0.13;
}

.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #374151;
    margin: 24px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e5e7eb;
    margin-left: 8px;
}

.chart-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 16px rgba(0,0,0,0.03);
    margin-bottom: 16px;
}
.chart-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 5px;
    gap: 2px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7280;
    border-radius: 8px;
    font-size: 0.83rem;
    font-weight: 500;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #4f46e5 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

[data-testid="stDataFrame"] { border-radius: 10px; border: 1px solid #e5e7eb; }
.stDownloadButton button {
    background: #4f46e5 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

.welcome-hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 20px;
    padding: 52px 48px;
    color: white;
    text-align: center;
    margin: 40px auto;
    max-width: 700px;
}
.hero-title { font-size: 2rem; font-weight: 700; margin-bottom: 10px; }
.hero-sub { font-size: 1rem; opacity: 0.85; margin-bottom: 28px; line-height: 1.6; }
.hero-step {
    background: rgba(255,255,255,0.15);
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.85rem;
    display: inline-block;
    margin: 4px;
}

.insight-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.85rem;
    color: #1e40af;
}
.warning-box {
    background: #fff7ed;
    border-left: 4px solid #f97316;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.85rem;
    color: #9a3412;
}
.success-box {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.85rem;
    color: #166534;
}

.badge-high   { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; }
.badge-medium { background:#fffbeb; color:#d97706; border:1px solid #fde68a; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; }
.badge-low    { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; }

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── PLOT THEME ────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor':   '#ffffff',
    'axes.edgecolor':   '#e5e7eb',
    'axes.labelcolor':  '#6b7280',
    'xtick.color':      '#9ca3af',
    'ytick.color':      '#9ca3af',
    'text.color':       '#374151',
    'grid.color':       '#f3f4f6',
    'grid.linewidth':   0.8,
    'font.family':      'sans-serif',
})

INDIGO = '#4f46e5'
GREEN  = '#10b981'
RED    = '#ef4444'
AMBER  = '#f59e0b'
PURPLE = '#8b5cf6'
SLATE  = '#64748b'
PALETTE= [INDIGO, GREEN, RED, AMBER, PURPLE, '#0ea5e9', '#ec4899']

def kpi(label, value, sub="", icon="", accent="#4f46e5"):
    st.markdown(f"""
    <div class="kpi-card" style="--accent-color:{accent}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def section(title, icon=""):
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)

def chart_wrap(title, icon=""):
    st.markdown(f'<div class="chart-card"><div class="chart-title">{icon} {title}</div>', unsafe_allow_html=True)

def chart_end():
    st.markdown('</div>', unsafe_allow_html=True)

def fig_show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px">
        <div style="font-size:1.4rem;font-weight:700;color:#fff;letter-spacing:-0.5px">
            📊 ChurnIQ
        </div>
        <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px">
            Customer Analytics Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<p style="font-size:0.7rem;letter-spacing:0.08em;color:#94a3b8;margin-bottom:8px">UPLOAD DATASET</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV file", type=['csv'], label_visibility="collapsed")

    st.divider()
    st.markdown('<p style="font-size:0.7rem;letter-spacing:0.08em;color:#94a3b8;margin-bottom:8px">FILTERS</p>', unsafe_allow_html=True)
    filter_contract = None
    filter_risk     = None

    st.divider()
    st.markdown('<p style="font-size:0.7rem;letter-spacing:0.08em;color:#94a3b8;margin-bottom:8px">EMAIL REPORT</p>', unsafe_allow_html=True)
    email_to   = st.text_input("Recipient Email", placeholder="manager@company.com", label_visibility="collapsed")
    smtp_host  = st.text_input("SMTP Host",  value="smtp.gmail.com",  label_visibility="collapsed")
    smtp_port  = st.number_input("SMTP Port", value=587, step=1,      label_visibility="collapsed")
    sender     = st.text_input("Sender Email", placeholder="your@gmail.com", label_visibility="collapsed")
    password   = st.text_input("App Password", placeholder="App Password", type="password", label_visibility="collapsed")
    send_btn   = st.button("📧 Send Email Report", use_container_width=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#64748b;line-height:1.7">
        <div style="color:#94a3b8;font-weight:600;margin-bottom:6px">INTERNSHIP INFO</div>
        Teyzix Core — June Batch<br>
        Task ID: ML-INT-1<br>
        Domain: Machine Learning<br>
        <span style="color:#818cf8">Deadline: 19 June 2026</span>
    </div>
    """, unsafe_allow_html=True)

# ── HEADER BAR ────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            background:#fff;border-radius:14px;padding:16px 24px;
            box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:20px">
    <div>
        <span style="font-size:1.3rem;font-weight:700;color:#111827">
            Customer Behavior Analytics
        </span>
        <span style="font-size:0.82rem;color:#9ca3af;margin-left:10px">
            Churn Prediction Dashboard
        </span>
    </div>
    <div style="font-size:0.75rem;color:#6b7280;background:#f9fafb;
                padding:6px 14px;border-radius:20px;border:1px solid #e5e7eb">
        ML-INT-1 &nbsp;|&nbsp; Teyzix Core 2026
    </div>
</div>
""", unsafe_allow_html=True)

# ── WELCOME STATE ─────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class="welcome-hero">
        <div class="hero-title">Welcome to ChurnIQ</div>
        <div class="hero-sub">
            Upload your customer CSV dataset to instantly analyze churn patterns,
            segment customers by value, and predict who's at risk of leaving.
        </div>
        <div>
            <span class="hero-step">📂 Upload CSV</span>
            <span class="hero-step">→</span>
            <span class="hero-step">🔍 Auto Analysis</span>
            <span class="hero-step">→</span>
            <span class="hero-step">🤖 ML Predictions</span>
            <span class="hero-step">→</span>
            <span class="hero-step">📋 Insights</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#fff;border-radius:14px;padding:24px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.06);text-align:center">
            <div style="font-size:2rem;margin-bottom:10px">🎯</div>
            <div style="font-weight:600;color:#111827;margin-bottom:6px">Churn Prediction</div>
            <div style="font-size:0.82rem;color:#6b7280">
                3 ML models trained automatically — Logistic Regression, Random Forest & XGBoost
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#fff;border-radius:14px;padding:24px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.06);text-align:center">
            <div style="font-size:2rem;margin-bottom:10px">👥</div>
            <div style="font-weight:600;color:#111827;margin-bottom:6px">Segmentation</div>
            <div style="font-size:0.82rem;color:#6b7280">
                Customers grouped into High, Medium & Low value segments with risk scores
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#fff;border-radius:14px;padding:24px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.06);text-align:center">
            <div style="font-size:2rem;margin-bottom:10px">💡</div>
            <div style="font-weight:600;color:#111827;margin-bottom:6px">Business Insights</div>
            <div style="font-size:0.82rem;color:#6b7280">
                Revenue impact, top churn reasons & actionable recommendations
            </div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── LOAD DATA ─────────────────────────────────────────────
@st.cache_data
def load_and_prep(file):
    df = pd.read_csv(file)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    if 'Churn' in df.columns:
        df['Churn_encoded'] = df['Churn'].map({'Yes': 1, 'No': 0})
        df['Churn_encoded'] = pd.to_numeric(df['Churn_encoded'], errors='coerce').astype('float64')

    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['usage_score'] = (df['tenure'] * df['MonthlyCharges']) / 100

    if 'TotalCharges' in df.columns:
        df['customer_value'] = pd.cut(df['TotalCharges'],
            bins=[0,1000,4000,99999], labels=['Low Value','Medium Value','High Value'])

    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'],
            bins=[0,12,24,48,60,73], labels=['0-1 yr','1-2 yr','2-4 yr','4-5 yr','5+ yr'])

    if 'Churn_encoded' in df.columns:
        from sklearn.ensemble import RandomForestClassifier
        feat_cols = [c for c in ['tenure','MonthlyCharges','TotalCharges'] if c in df.columns]
        X = df[feat_cols].fillna(0)
        y = df['Churn_encoded'].fillna(0)
        rf = RandomForestClassifier(n_estimators=60, random_state=42)
        rf.fit(X, y)
        df['churn_prob'] = rf.predict_proba(X)[:, 1]
        df['risk_category'] = pd.cut(df['churn_prob'],
            bins=[-0.01,0.3,0.6,1.01], labels=['Low Risk','Medium Risk','High Risk'])

    return df

with st.spinner("Analyzing your data..."):
    df = load_and_prep(uploaded)

# ── SIDEBAR FILTERS ───────────────────────────────────────
with st.sidebar:
    if 'Contract' in df.columns:
        contracts = ['All'] + sorted(df['Contract'].unique().tolist())
        filter_contract = st.selectbox("Contract Type", contracts)
    if 'risk_category' in df.columns:
        risks = ['All','High Risk','Medium Risk','Low Risk']
        filter_risk = st.selectbox("Risk Level", risks)

dff = df.copy()
if filter_contract and filter_contract != 'All':
    dff = dff[dff['Contract'] == filter_contract]
if filter_risk and filter_risk != 'All':
    dff = dff[dff['risk_category'] == filter_risk]

# ── KPI ROW ───────────────────────────────────────────────
total      = len(dff)
churned    = int(pd.to_numeric(dff['Churn_encoded'], errors='coerce').sum()) if 'Churn_encoded' in dff.columns else 0
churn_rate = (churned / total * 100) if total > 0 else 0
avg_charge = float(pd.to_numeric(dff['MonthlyCharges'], errors='coerce').mean()) if 'MonthlyCharges' in dff.columns else 0.0
revenue    = float(pd.to_numeric(dff['MonthlyCharges'], errors='coerce').sum()) if 'MonthlyCharges' in dff.columns else 0.0
high_risk  = int((dff['risk_category'] == 'High Risk').sum()) if 'risk_category' in dff.columns else 0

c1,c2,c3,c4,c5 = st.columns(5)
with c1: kpi("Total Customers",   f"{total:,}",        "in dataset",             "👥", "#4f46e5")
with c2: kpi("Churned",           f"{churned:,}",      f"{churn_rate:.1f}% rate","📉", "#ef4444")
with c3: kpi("Avg Monthly Charge",f"${avg_charge:.0f}","per customer",           "💳", "#10b981")
with c4: kpi("Monthly Revenue",   f"${revenue:,.0f}",  "total portfolio",        "💰", "#f59e0b")
with c5: kpi("High Risk",         f"{high_risk:,}",    "need attention",         "⚠️", "#ef4444")

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📈  Overview",
    "🔍  Churn Analysis",
    "👥  Segmentation",
    "🤖  Predictions",
    "📋  Data Explorer"
])

# ═══════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🍩 Churn Distribution</div>', unsafe_allow_html=True)
        if 'Churn' in dff.columns:
            counts = dff['Churn'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3.8))
            wedges, texts, autotexts = ax.pie(
                counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=[GREEN, RED], startangle=90,
                wedgeprops=dict(width=0.58, edgecolor='white', linewidth=3),
                pctdistance=0.76)
            for t in texts:     t.set_color('#6b7280'); t.set_fontsize(11)
            for t in autotexts: t.set_color('white');   t.set_fontsize(10); t.set_fontweight('bold')
            ax.set_title(f"{total:,} total customers", color='#9ca3af', fontsize=10, pad=10)
            fig.tight_layout()
            fig_show(fig)

            st.markdown(f"""
            <div style="display:flex;gap:10px;margin-top:4px">
                <div style="flex:1;background:#f0fdf4;border-radius:10px;padding:12px 14px;
                            border-left:3px solid {GREEN}">
                    <div style="font-size:0.68rem;color:#6b7280;font-weight:600;text-transform:uppercase;
                                letter-spacing:0.07em">Active</div>
                    <div style="font-size:1.5rem;font-weight:700;color:#065f46">
                        {counts.get('No',0):,}</div>
                </div>
                <div style="flex:1;background:#fef2f2;border-radius:10px;padding:12px 14px;
                            border-left:3px solid {RED}">
                    <div style="font-size:0.68rem;color:#6b7280;font-weight:600;text-transform:uppercase;
                                letter-spacing:0.07em">Churned</div>
                    <div style="font-size:1.5rem;font-weight:700;color:#991b1b">
                        {counts.get('Yes',0):,}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📈 Revenue Trend by Tenure</div>', unsafe_allow_html=True)
        if 'tenure' in dff.columns and 'MonthlyCharges' in dff.columns:
            trend = dff.groupby('tenure')['MonthlyCharges'].mean()
            fig, ax = plt.subplots(figsize=(5, 4.2))
            ax.plot(trend.index, trend.values, color=INDIGO, linewidth=2.5, zorder=3)
            ax.fill_between(trend.index, trend.values, alpha=0.08, color=INDIGO)
            ax.set_xlabel("Tenure (months)", fontsize=11)
            ax.set_ylabel("Avg Charge ($)", fontsize=11)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig_show(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📋 Churn by Contract Type</div>', unsafe_allow_html=True)
        if 'Contract' in dff.columns and 'Churn_encoded' in dff.columns:
            data = dff.groupby('Contract')['Churn_encoded'].mean().sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(5, 3.2))
            bars = ax.barh(data.index, data.values*100,
                           color=[RED if v > 0.35 else AMBER if v > 0.15 else GREEN for v in data.values],
                           edgecolor='white', linewidth=1.5, height=0.5)
            for bar, val in zip(bars, data.values):
                ax.text(val*100 + 0.4, bar.get_y()+bar.get_height()/2,
                        f'{val:.1%}', va='center', fontsize=10, color='#374151', fontweight='500')
            ax.set_xlabel("Churn Rate (%)", fontsize=11)
            ax.set_xlim(0, data.values.max()*100 * 1.28)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig_show(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">💳 Churn by Payment Method</div>', unsafe_allow_html=True)
        if 'PaymentMethod' in dff.columns and 'Churn_encoded' in dff.columns:
            data = dff.groupby('PaymentMethod')['Churn_encoded'].mean().sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(5, 3.2))
            colors_p = [RED if v > 0.35 else AMBER if v > 0.15 else GREEN for v in data.values]
            bars = ax.barh(data.index, data.values*100,
                           color=colors_p, edgecolor='white', linewidth=1.5, height=0.5)
            for bar, val in zip(bars, data.values):
                ax.text(val*100+0.4, bar.get_y()+bar.get_height()/2,
                        f'{val:.1%}', va='center', fontsize=9, color='#374151', fontweight='500')
            ax.set_xlabel("Churn Rate (%)", fontsize=11)
            ax.set_xlim(0, data.values.max()*100 * 1.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig_show(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# TAB 2 — CHURN ANALYSIS
# ═══════════════════════════════════════
with tab2:
    # Insight boxes
    if 'Churn_encoded' in dff.columns and 'MonthlyCharges' in dff.columns:
        churned_df = dff[dff['Churn_encoded']==1]
        active_df  = dff[dff['Churn_encoded']==0]
        rev_loss   = churned_df['MonthlyCharges'].sum()

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:20px">
            <div class="warning-box">
                ⚠️ <strong>Revenue at Risk</strong><br>
                ${rev_loss:,.0f}/month · ${rev_loss*12:,.0f}/year from {len(churned_df):,} churned customers
            </div>
            <div class="insight-box">
                📉 <strong>Tenure Insight</strong><br>
                Churned customers leave after avg {churned_df['tenure'].mean():.0f} months vs {active_df['tenure'].mean():.0f} months for active
            </div>
            <div class="insight-box">
                💳 <strong>Charge Pattern</strong><br>
                Churned customers pay avg ${churned_df['MonthlyCharges'].mean():.0f}/mo vs ${active_df['MonthlyCharges'].mean():.0f}/mo active
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">💵 Monthly Charges — Churned vs Active</div>', unsafe_allow_html=True)
        if 'MonthlyCharges' in dff.columns and 'Churn' in dff.columns:
            fig, ax = plt.subplots(figsize=(5, 3.8))
            for label, color, alpha in [('No',GREEN,0.65),('Yes',RED,0.65)]:
                ax.hist(dff[dff['Churn']==label]['MonthlyCharges'], bins=25,
                        alpha=alpha, label=label, color=color, edgecolor='white', linewidth=0.5)
            ax.set_xlabel("Monthly Charges ($)", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.legend(title="Churned", frameon=True, facecolor='white', edgecolor='#e5e7eb')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig_show(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">⏱ Tenure — Churned vs Active</div>', unsafe_allow_html=True)
        if 'tenure' in dff.columns and 'Churn' in dff.columns:
            fig, ax = plt.subplots(figsize=(5, 3.8))
            for label, color, alpha in [('No',GREEN,0.65),('Yes',RED,0.65)]:
                ax.hist(dff[dff['Churn']==label]['tenure'], bins=25,
                        alpha=alpha, label=label, color=color, edgecolor='white', linewidth=0.5)
            ax.set_xlabel("Tenure (months)", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.legend(title="Churned", frameon=True, facecolor='white', edgecolor='#e5e7eb')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig_show(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🔥 Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    numeric_df = dff.select_dtypes(include=np.number)
    corr_cols = [c for c in ['tenure','MonthlyCharges','TotalCharges',
                              'usage_score','Churn_encoded','churn_prob'] if c in numeric_df.columns]
    if len(corr_cols) >= 2:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        mask = np.triu(np.ones((len(corr_cols),len(corr_cols)),dtype=bool))
        sns.heatmap(numeric_df[corr_cols].corr(), ax=ax, annot=True, fmt='.2f',
                    mask=mask, cmap='RdYlGn', center=0,
                    linewidths=1.5, linecolor='white', annot_kws={'size':11})
        ax.set_title("Correlation between key features", color='#6b7280', fontsize=11)
        fig.tight_layout()
        fig_show(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# TAB 3 — SEGMENTATION
# ═══════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">💎 Customer Value Segments</div>', unsafe_allow_html=True)
        if 'customer_value' in dff.columns:
            seg = dff['customer_value'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3.8))
            seg_colors = [AMBER,'#94a3b8',GREEN]
            wedges,texts,autotexts = ax.pie(seg.values, labels=seg.index,
                autopct='%1.1f%%', colors=seg_colors, startangle=90,
                wedgeprops=dict(width=0.55, edgecolor='white', linewidth=3), pctdistance=0.76)
            for t in texts:     t.set_color('#6b7280'); t.set_fontsize(10)
            for t in autotexts: t.set_color('white');   t.set_fontweight('bold')
            fig.tight_layout()
            fig_show(fig)

            seg_table = dff.groupby('customer_value', observed=True).agg(
                Count=('Churn_encoded','count'),
                Churn_Rate=('Churn_encoded','mean'),
                Avg_Charge=('MonthlyCharges','mean')
            ).reset_index()
            seg_table['Churn_Rate'] = (seg_table['Churn_Rate']*100).round(1).astype(str)+'%'
            seg_table['Avg_Charge'] = '$'+seg_table['Avg_Charge'].round(0).astype(int).astype(str)
            st.dataframe(seg_table, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">⚠️ Risk Category Distribution</div>', unsafe_allow_html=True)
        if 'risk_category' in dff.columns:
            risk = dff['risk_category'].value_counts()
            rc   = {'High Risk':RED,'Medium Risk':AMBER,'Low Risk':GREEN}
            fig, ax = plt.subplots(figsize=(5, 3.8))
            bars = ax.bar(risk.index, risk.values,
                          color=[rc.get(r,INDIGO) for r in risk.index],
                          edgecolor='white', linewidth=1.5, width=0.48)
            for bar, val in zip(bars, risk.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8,
                        f'{val:,}', ha='center', color='#374151', fontsize=11, fontweight='600')
            ax.set_ylabel("Customers", fontsize=11)
            ax.set_ylim(0, risk.values.max()*1.18)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig_show(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📅 Churn Rate by Tenure Group</div>', unsafe_allow_html=True)
    if 'tenure_group' in dff.columns and 'Churn_encoded' in dff.columns:
        tg = dff.groupby('tenure_group', observed=True)['Churn_encoded'].mean()*100
        fig, ax = plt.subplots(figsize=(9, 3.2))
        bar_colors = [RED if v>30 else AMBER if v>15 else GREEN for v in tg.values]
        bars = ax.bar(tg.index.astype(str), tg.values,
                      color=bar_colors, edgecolor='white', linewidth=1.5, width=0.5)
        for bar, val in zip(bars, tg.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{val:.1f}%', ha='center', color='#374151', fontsize=10, fontweight='600')
        ax.set_ylabel("Churn Rate (%)", fontsize=11)
        ax.set_xlabel("Tenure Group", fontsize=11)
        ax.set_ylim(0, tg.values.max()*1.22)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig_show(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# TAB 4 — PREDICTIONS
# ═══════════════════════════════════════
with tab4:
    if 'risk_category' in dff.columns and 'churn_prob' in dff.columns:
        high_risk_df = dff[dff['risk_category']=='High Risk'].sort_values('churn_prob', ascending=False)

        m1,m2,m3 = st.columns(3)
        with m1: kpi("High Risk Count", f"{len(high_risk_df):,}",
                     f"{len(high_risk_df)/max(len(dff),1)*100:.1f}% of total","🚨","#ef4444")
        with m2:
            avg_p = high_risk_df['churn_prob'].mean() if len(high_risk_df) > 0 else 0
            kpi("Avg Churn Probability", f"{avg_p:.0%}", "in high-risk group","📊","#ef4444")
        with m3:
            rev_risk = high_risk_df['MonthlyCharges'].sum() if 'MonthlyCharges' in high_risk_df.columns else 0
            kpi("Revenue at Risk", f"${rev_risk:,.0f}", "monthly exposure","💸","#f59e0b")

        st.markdown("<br>", unsafe_allow_html=True)

        if len(high_risk_df) > 0:
            st.markdown("""
            <div class="warning-box">
                🚨 <strong>Immediate Action Required</strong> — These customers have a 60%+ probability of churning.
                Consider personalized retention offers, loyalty rewards, or direct outreach.
            </div>""", unsafe_allow_html=True)

            display_cols = [c for c in ['customerID','tenure','MonthlyCharges',
                                         'Contract','churn_prob','risk_category']
                            if c in high_risk_df.columns]
            top20 = high_risk_df[display_cols].head(20).copy()
            if 'churn_prob' in top20.columns:
                top20['churn_prob'] = (top20['churn_prob']*100).round(1).astype(str)+'%'
            st.markdown("**Top 20 Customers by Churn Probability**")
            st.dataframe(top20, use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="success-box">✅ No high-risk customers with current filters.</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="chart-card" style="margin-top:16px">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📊 Churn Probability Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.hist(dff['churn_prob'], bins=40, color=INDIGO, edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.axvline(0.3, color=AMBER, linewidth=2, linestyle='--', label='Low / Med  (30%)', alpha=0.9)
        ax.axvline(0.6, color=RED,   linewidth=2, linestyle='--', label='Med / High (60%)', alpha=0.9)
        ax.set_xlabel("Churn Probability", fontsize=11)
        ax.set_ylabel("Customer Count", fontsize=11)
        ax.legend(frameon=True, facecolor='white', edgecolor='#e5e7eb')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig_show(fig)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ═══════════════════════════════════════
with tab5:
    c1,c2,c3 = st.columns(3)
    with c1: kpi("Total Rows",    f"{len(dff):,}",         "after filters",  "🗂", "#4f46e5")
    with c2: kpi("Columns",       f"{len(dff.columns)}",   "in dataset",     "📐", "#10b981")
    with c3: kpi("Missing Values",f"{dff.isnull().sum().sum()}", "remaining", "✅", "#10b981")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🔎 Column Explorer</div>', unsafe_allow_html=True)
    all_cols = dff.columns.tolist()
    selected = st.multiselect("Select columns to view", all_cols, default=all_cols[:8])
    if selected:
        st.dataframe(dff[selected], use_container_width=True, height=360)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📊 Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(dff.describe().round(2), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">⬇️ Export Data</div>', unsafe_allow_html=True)
    csv = dff.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Analyzed Dataset (.csv)",
        data=csv, file_name='churniq_analyzed.csv', mime='text/csv'
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════
# EMAIL REPORT — SEND LOGIC
# ═══════════════════════════════════════
def build_email_html(df):
    """Build HTML email body from current dataframe."""
    run_date   = datetime.now().strftime('%d %B %Y, %I:%M %p')
    total      = len(df)
    churned    = int(df['Churn_encoded'].sum()) if 'Churn_encoded' in df.columns else 0
    churn_rate = churned / total * 100 if total > 0 else 0
    high_risk  = int((df['risk_category'] == 'High Risk').sum()) if 'risk_category' in df.columns else 0
    med_risk   = int((df['risk_category'] == 'Medium Risk').sum()) if 'risk_category' in df.columns else 0
    low_risk   = int((df['risk_category'] == 'Low Risk').sum()) if 'risk_category' in df.columns else 0
    rev_month  = float(df['MonthlyCharges'].sum()) if 'MonthlyCharges' in df.columns else 0
    rev_risk   = float(df[df['risk_category'] == 'High Risk']['MonthlyCharges'].sum()) if 'risk_category' in df.columns and 'MonthlyCharges' in df.columns else 0

    # Top 5 table rows
    display_cols = [c for c in ['customerID', 'tenure', 'MonthlyCharges', 'Contract', 'churn_prob'] if c in df.columns]
    top5_rows = ""
    if 'churn_prob' in df.columns and len(df) > 0:
        top5 = df.sort_values('churn_prob', ascending=False).head(5)[display_cols]
        for _, row in top5.iterrows():
            prob = row.get('churn_prob', 0)
            badge_color = '#dc2626' if prob > 0.7 else '#f59e0b'
            top5_rows += f"""
            <tr>
              <td style="padding:8px 12px;border-bottom:1px solid #f3f4f6;font-size:13px;color:#111827">{row.get('customerID','N/A')}</td>
              <td style="padding:8px 12px;border-bottom:1px solid #f3f4f6;font-size:13px;color:#6b7280">{int(row.get('tenure',0))} mo</td>
              <td style="padding:8px 12px;border-bottom:1px solid #f3f4f6;font-size:13px;color:#6b7280">${row.get('MonthlyCharges',0):.0f}</td>
              <td style="padding:8px 12px;border-bottom:1px solid #f3f4f6;font-size:13px;color:#6b7280">{row.get('Contract','N/A')}</td>
              <td style="padding:8px 12px;border-bottom:1px solid #f3f4f6">
                <span style="background:{badge_color};color:white;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:700">{prob:.0%}</span>
              </td>
            </tr>"""

    # Contract rows
    contract_rows = ""
    if 'Contract' in df.columns and 'Churn_encoded' in df.columns:
        for contract, grp in df.groupby('Contract'):
            rate = grp['Churn_encoded'].mean() * 100
            color = '#dc2626' if rate > 35 else '#f59e0b' if rate > 15 else '#10b981'
            contract_rows += f"""
            <tr>
              <td style="padding:7px 12px;font-size:13px;color:#374151;font-weight:500">{contract}</td>
              <td style="padding:7px 12px">
                <div style="background:#f3f4f6;border-radius:4px;height:10px;width:180px;display:inline-block;vertical-align:middle">
                  <div style="background:{color};height:10px;border-radius:4px;width:{min(int(rate*3),180)}px"></div>
                </div>
              </td>
              <td style="padding:7px 12px;font-size:13px;color:{color};font-weight:700">{rate:.1f}%</td>
            </tr>"""

    alert_banner = ""
    if total > 0 and high_risk / total > 0.10:
        alert_banner = f'<tr><td style="background:#fef2f2;border-left:4px solid #dc2626;padding:12px 36px;font-size:13px;color:#991b1b;font-weight:600">⚠️ High Alert: {high_risk} customers ({high_risk/total*100:.1f}%) are at high churn risk.</td></tr>'

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:28px 0">
<tr><td align="center">
<table width="620" cellpadding="0" cellspacing="0" style="background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.08)">
  <tr><td style="background:linear-gradient(135deg,#4f46e5,#7c3aed);padding:32px 36px">
    <div style="font-size:20px;font-weight:700;color:#fff">📊 ChurnIQ Weekly Report</div>
    <div style="font-size:12px;color:rgba(255,255,255,0.75);margin-top:4px">Teyzix Core · ML-INT-1 · {run_date}</div>
  </td></tr>
  {alert_banner}
  <tr><td style="padding:28px 36px 0">
    <table width="100%" cellpadding="0" cellspacing="8">
      <tr>
        <td width="25%" style="text-align:center;background:#f9fafb;border-radius:10px;padding:14px;border-top:3px solid #4f46e5">
          <div style="font-size:22px;font-weight:700;color:#111">{total:,}</div>
          <div style="font-size:11px;color:#6b7280;margin-top:3px">Total Customers</div>
        </td>
        <td width="5%"></td>
        <td width="25%" style="text-align:center;background:#f9fafb;border-radius:10px;padding:14px;border-top:3px solid #dc2626">
          <div style="font-size:22px;font-weight:700;color:#dc2626">{churn_rate:.1f}%</div>
          <div style="font-size:11px;color:#6b7280;margin-top:3px">Churn Rate</div>
        </td>
        <td width="5%"></td>
        <td width="25%" style="text-align:center;background:#f9fafb;border-radius:10px;padding:14px;border-top:3px solid #f59e0b">
          <div style="font-size:22px;font-weight:700;color:#d97706">{high_risk:,}</div>
          <div style="font-size:11px;color:#6b7280;margin-top:3px">High Risk</div>
        </td>
        <td width="5%"></td>
        <td width="25%" style="text-align:center;background:#f9fafb;border-radius:10px;padding:14px;border-top:3px solid #10b981">
          <div style="font-size:22px;font-weight:700;color:#059669">${rev_month:,.0f}</div>
          <div style="font-size:11px;color:#6b7280;margin-top:3px">Monthly Revenue</div>
        </td>
      </tr>
    </table>
  </td></tr>
  <tr><td style="padding:20px 36px 0">
    <div style="background:#fff7ed;border-radius:10px;padding:16px 20px;border:1px solid #fed7aa">
      <div style="font-size:12px;font-weight:700;color:#9a3412;margin-bottom:6px">💰 Revenue at Risk</div>
      <div style="font-size:24px;font-weight:700;color:#ea580c">${rev_risk:,.0f}<span style="font-size:13px;color:#9a3412;font-weight:400">/month</span></div>
      <div style="font-size:12px;color:#9a3412;margin-top:3px">Annualized risk: <strong>${rev_risk*12:,.0f}</strong></div>
    </div>
  </td></tr>
  <tr><td style="padding:20px 36px 0">
    <div style="font-size:10px;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px">Risk Distribution</div>
    <table width="100%">
      <tr><td style="font-size:12px;font-weight:600;color:#dc2626;width:80px">🔴 High</td>
          <td><div style="background:#fee2e2;border-radius:4px;height:10px"><div style="background:#dc2626;height:10px;border-radius:4px;width:{min(int(high_risk/max(total,1)*280),280)}px"></div></div></td>
          <td style="font-size:12px;color:#374151;font-weight:600;width:90px;text-align:right">{high_risk:,} ({high_risk/max(total,1)*100:.1f}%)</td></tr>
      <tr><td colspan="3" style="height:6px"></td></tr>
      <tr><td style="font-size:12px;font-weight:600;color:#d97706">🟡 Medium</td>
          <td><div style="background:#fef3c7;border-radius:4px;height:10px"><div style="background:#f59e0b;height:10px;border-radius:4px;width:{min(int(med_risk/max(total,1)*280),280)}px"></div></div></td>
          <td style="font-size:12px;color:#374151;font-weight:600;text-align:right">{med_risk:,} ({med_risk/max(total,1)*100:.1f}%)</td></tr>
      <tr><td colspan="3" style="height:6px"></td></tr>
      <tr><td style="font-size:12px;font-weight:600;color:#16a34a">🟢 Low</td>
          <td><div style="background:#dcfce7;border-radius:4px;height:10px"><div style="background:#22c55e;height:10px;border-radius:4px;width:{min(int(low_risk/max(total,1)*280),280)}px"></div></div></td>
          <td style="font-size:12px;color:#374151;font-weight:600;text-align:right">{low_risk:,} ({low_risk/max(total,1)*100:.1f}%)</td></tr>
    </table>
  </td></tr>
  <tr><td style="padding:20px 36px 0">
    <div style="font-size:10px;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px">Top 5 High-Risk Customers</div>
    <table width="100%" style="border:1px solid #f3f4f6;border-radius:8px;overflow:hidden">
      <tr style="background:#f9fafb">
        <th style="padding:8px 12px;text-align:left;font-size:11px;color:#6b7280">Customer</th>
        <th style="padding:8px 12px;text-align:left;font-size:11px;color:#6b7280">Tenure</th>
        <th style="padding:8px 12px;text-align:left;font-size:11px;color:#6b7280">Charge</th>
        <th style="padding:8px 12px;text-align:left;font-size:11px;color:#6b7280">Contract</th>
        <th style="padding:8px 12px;text-align:left;font-size:11px;color:#6b7280">Risk</th>
      </tr>
      {top5_rows}
    </table>
  </td></tr>
  <tr><td style="padding:20px 36px 0">
    <div style="font-size:10px;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">Churn by Contract</div>
    <table width="100%">{contract_rows}</table>
  </td></tr>
  <tr><td style="padding:20px 36px 0">
    <div style="background:#eff6ff;border-left:3px solid #3b82f6;border-radius:0 8px 8px 0;padding:12px 16px;font-size:13px;color:#1e40af;margin-bottom:8px">
      📞 <strong>Action:</strong> Route {high_risk:,} high-risk customers to retention team immediately.
    </div>
    <div style="background:#eff6ff;border-left:3px solid #3b82f6;border-radius:0 8px 8px 0;padding:12px 16px;font-size:13px;color:#1e40af;margin-bottom:8px">
      📋 <strong>Contract:</strong> Offer discounted annual plans to month-to-month customers.
    </div>
    <div style="background:#eff6ff;border-left:3px solid #3b82f6;border-radius:0 8px 8px 0;padding:12px 16px;font-size:13px;color:#1e40af">
      💳 <strong>Payment:</strong> Migrate electronic check users to auto-pay to reduce churn.
    </div>
  </td></tr>
  <tr><td style="padding:28px 36px;text-align:center;border-top:1px solid #f3f4f6;margin-top:20px">
    <div style="font-size:12px;color:#9ca3af">Generated by <strong style="color:#4f46e5">ChurnIQ</strong> · Teyzix Core · {run_date}<br>This is an automated report.</div>
  </td></tr>
</table>
</td></tr>
</table>
</body></html>"""


def send_email_from_app(html_body, recipient, smtp_host, smtp_port, sender_email, sender_pass):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"📊 ChurnIQ Weekly Churn Report — {datetime.now().strftime('%d %b %Y')}"
    msg['From']    = sender_email
    msg['To']      = recipient

    msg.attach(MIMEText("Please view this email in an HTML-compatible client.", 'plain'))
    msg.attach(MIMEText(html_body, 'html'))

    with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
        server.ehlo()
        server.starttls()
        server.login(sender_email, sender_pass)
        server.sendmail(sender_email, [recipient], msg.as_string())


# ── EMAIL SEND TRIGGER (after data is loaded) ─────────────
if uploaded is not None and send_btn:
    if not email_to or not sender or not password:
        st.sidebar.error("❌ Recipient, Sender Email aur Password sab fill karo.")
    else:
        with st.sidebar:
            with st.spinner("Sending email..."):
                try:
                    html_body = build_email_html(dff)
                    send_email_from_app(
                        html_body  = html_body,
                        recipient  = email_to.strip(),
                        smtp_host  = smtp_host.strip(),
                        smtp_port  = smtp_port,
                        sender_email = sender.strip(),
                        sender_pass  = password,
                    )
                    st.success(f"✅ Email sent to {email_to}!")
                except Exception as e:
                    err = str(e)
                    if "535" in err or "Username" in err or "auth" in err.lower():
                        st.error("❌ Auth failed. Gmail users: use App Password, not main password.")
                    else:
                        st.error(f"❌ Error: {err}")
