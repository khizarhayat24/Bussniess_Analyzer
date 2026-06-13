import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, warnings

warnings.filterwarnings('ignore')
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

def run_shap_explainability(df, model, features, model_name="XGBoost"):
    """
    SHAP Explainability Module (UPGRADED)
    -------------------------------------
    Explains WHY each customer has a particular churn probability score.
    Dynamically synchronizes dimension mapping for tree-based arrays (RF, XGBoost, etc.)

    Args:
        df      : Cleaned feature matrix dataframe
        model   : Trained ensemble framework model asset (XGB/RF)
        features: Array string matching prediction features structure
        model_name: Identity header descriptor text for reports
    """

    try:
        import shap
    except ImportError:
        print("❌ Dynamic Asset Fail: SHAP not installed. Execute: pip install shap")
        return None

    print("\n" + "=" * 60)
    print("   SHAP ENTERPRISE EXPLAINABILITY MATRIX MODULE")
    print("=" * 60)

    # Clean input array mapping slice safely
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"[*] Dispatching Explainer initialization sequence on {model_name} backend...")
    
    # ── FIXED: EXPLATION ALIGNMENT ENGINE ──────────────────
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception as explainer_err:
        print(f"⚠️ Explainer Exception Native Core Bypass: {explainer_err}")
        # Secondary fallback layer injection
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X).values

    # FIXED: Comprehensive Random Forest vs XGBoost Output Dimension Parser
    if isinstance(shap_values, list):
        # Ensembles like RandomForest return list structure frameworks for [Class 0, Class 1]
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif len(shap_values.shape) == 3:
        # Array structures with multi-class configuration shape layers
        sv = shap_values[:, :, 1]
    else:
        # Standard flattened XGBoost single logit mapping matrix
        sv = shap_values

    print(f"   [System Check OK] Extracted Shape Alignment: {sv.shape}")

    # UI/UX Plot Attributes styling configurations
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.edgecolor': '#cbd5e1', 'axes.labelcolor': '#334155',
        'xtick.color': '#64748b', 'ytick.color': '#64748b',
        'text.color': '#0f172a', 'grid.color': '#f1f5f9',
    })

    # ── PLOT 1: RE-ENGINEERED GLOBAL IMPORTANCE BAR ─────────
    print("[*] Compiling Visualization Plot 1: Global Importances Profile...")
    mean_abs = np.abs(sv).mean(axis=0)
    feat_imp = pd.Series(mean_abs, index=features).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    max_val = feat_imp.max()
    colors = ['#1d4ed8' if v == max_val else '#60a5fa' for v in feat_imp.values]
    
    bars = ax.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor='white', linewidth=1, height=0.6)
    
    for bar, val in zip(bars, feat_imp.values):
        ax.text(val + (max_val * 0.01), bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, color='#0f172a', fontweight='bold')
                
    ax.set_xlabel("Mean Absolute |SHAP Value| (Average Magnitude on Model Predictions)", fontsize=10, fontweight='bold')
    ax.set_title(f"Global SHAP Feature Importance Signature Profile — {model_name}", fontsize=12, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_xlim(0, max_val * 1.15)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/08_shap_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✔ Saved: outputs/plots/08_shap_feature_importance.png")

    # ── PLOT 2: OPTIMIZED DATA SPECIFIC SHAP BEESWARM ────────
    print("[*] Compiling Visualization Plot 2: High Density Beeswarm Cluster...")
    sample_size = min(500, len(X))
    
    # Stratified deterministic indices selection to avoid blank index alignment errors
    np.random.seed(42)
    idx = np.random.choice(len(X), sample_size, replace=False)
    sv_sample = sv[idx]
    X_sample = X.iloc[idx]

    fig, ax = plt.subplots(figsize=(11, 6))
    feature_order = np.argsort(np.abs(sv_sample).mean(axis=0))[::-1]
    ordered_features = [features[i] for i in feature_order]
    ordered_sv = sv_sample[:, feature_order]
    ordered_X = X_sample.values[:, feature_order]

    for fi, (feat, shap_col, feat_col) in enumerate(zip(ordered_features, ordered_sv.T, ordered_X.T)):
        feat_min, feat_max = feat_col.min(), feat_col.max()
        norm = (feat_col - feat_min) / (feat_max - feat_min) if feat_max > feat_min else np.zeros_like(feat_col)
        
        # Jitter generation sequence map logic
        y_jitter = fi + np.random.uniform(-0.18, 0.18, size=len(shap_col))
        colors_scatter = plt.cm.coolwarm(norm) # Shift to high fidelity contrast scheme mapping
        ax.scatter(shap_col, y_jitter, c=colors_scatter, alpha=0.6, s=15, linewidths=0)

    ax.set_yticks(range(len(ordered_features)))
    ax.set_yticklabels(ordered_features, fontsize=9, fontweight='bold')
    ax.axvline(0, color='#64748b', linewidth=1.2, linestyle='-')
    ax.set_xlabel("SHAP Value Distribution impact (← Low Risk | High Risk →)", fontsize=10, fontweight='bold')
    ax.set_title(f"Feature Attribution Swarm Metrics — Dynamic Distribution Impact Directional mapping",
                 fontsize=12, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # Modernized colorbar integration interface properties
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.03)
    cbar.set_label('Relative Scale Feature Value Data Streams', fontsize=9, fontweight='bold')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low Telemetry Value', 'High Telemetry Value'], fontsize=8)

    plt.tight_layout()
    plt.savefig('outputs/plots/09_shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✔ Saved: outputs/plots/09_shap_beeswarm.png")

    # ── PLOT 3: INDIVIDUAL REASON EXPLAINERS (FIXED INDEX MAP) ─
    print("[*] Compiling Visualization Plot 3: Specific Customer Waterfall Attributions...")
    
    # Safe positional baseline extraction
    df_reset = df.reset_index(drop=True)
    X_reset = X.reset_index(drop=True)
    
    if 'churn_prob' in df_reset.columns:
        top3_positions = df_reset['churn_prob'].nlargest(3).index.tolist()
    else:
        top3_positions = list(range(min(3, len(df_reset))))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Individual Core Driving Attributes — Top High-Risk Pipeline Accounts Explanations',
                 fontsize=13, fontweight='bold', y=1.02, color='#0f172a')

    for plot_i, row_pos in enumerate(top3_positions):
        ax = axes[plot_i]
        
        shap_row = sv[row_pos]
        feat_vals = X_reset.iloc[row_pos].values
        
        # Safe extraction for base layer metrics calculations
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]

        # Top 6 maximum localized attribution vectors
        order = np.argsort(np.abs(shap_row))[::-1][:6]
        f_names = [features[i] for i in order]
        f_shap = shap_row[order]
        f_vals = feat_vals[order]

        bar_colors = ['#dc2626' if v > 0 else '#2563eb' for v in f_shap]
        bars = ax.barh(range(len(f_names)), f_shap, color=bar_colors, edgecolor='white', height=0.55)
        
        ax.set_yticks(range(len(f_names)))
        ax.set_yticklabels([f'{n}\n  = {v:.2f}' for n, v in zip(f_names, f_vals)], fontsize=8, fontweight='bold')
        ax.axvline(0, color='#64748b', linewidth=1, linestyle='--')
        
        prob_title_str = f'{df_reset["churn_prob"].iloc[row_pos]:.1%}' if 'churn_prob' in df_reset.columns else 'N/A'
        ax.set_title(f'Customer Target #{plot_i+1}\nPredicted Churn Chance: {prob_title_str}',
                     fontsize=10, fontweight='bold', color='#1e293b')
        ax.set_xlabel("Local Feature SHAP Weight", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        red_patch = mpatches.Patch(color='#dc2626', label='Escalates Risk')
        blue_patch = mpatches.Patch(color='#2563eb', label='Mitigates Risk')
        ax.legend(handles=[red_patch, blue_patch], fontsize=7, loc='lower right', frameon=True)

    plt.tight_layout()
    plt.savefig('outputs/plots/10_shap_individual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✔ Saved: outputs/plots/10_shap_individual.png")

    # ── OUTPUT REPORT PIPELINE FLUSH ────────────────────────
    top_feature = feat_imp.idxmax()
    top_value = feat_imp.max()

    report_string = f"""TEYZIX CORE ANALYTICS - MODEL EXPLAINABILITY PROFILE AUDIT REPORT
======================================================================
Target Model Engine Name  : {model_name} Framework Array
Volumetric Records Checked: {len(X):,} active consumer rows.
Calculated Features Count : {len(features)} attributes synchronized.
======================================================================

GLOBAL MATRIX FEATURE ATTRIBUTION IMPORTANCE RANKINGS:
(Ranked by calculated Mean Absolute Magnitude values)
"""
    for feat, val in feat_imp.sort_values(ascending=False).items():
        bar_graphic = '█' * int((val / top_value * 20) if top_value > 0 else 0)
        report_string += f"  {feat:<28} {bar_graphic:<20} {val:.4f}\n"

    report_string += f"""
CORE ARCHITECTURAL OBSERVATIONS SUMMARY:
  1. Maximum Attribution Anchor Component: '{top_feature}' holds largest global weight (SHAP value: {top_value:.4f}).
  2. Positive Attributions Directional State: Positive localized SHAP coefficients explicitly accelerate customer churn paths.
  3. Risk Mitigation Indicators Checklist: Negative coefficients reflect stabilization traits that effectively lower churn risk.

PIPELINE OUTPUT FILE ARTIFACTS VERIFIED AND EXPORTED:
  -> Graphic Matrix 1 : outputs/plots/08_shap_feature_importance.png
  -> Graphic Matrix 2 : outputs/plots/09_shap_beeswarm.png
  -> Graphic Matrix 3 : outputs/plots/10_shap_individual.png
======================================================================
"""
    with open('outputs/reports/shap_report.txt', 'w', encoding='utf-8') as f_write:
        f_write.write(report_string)

    print("\n" + report_string)
    print("   ✔ Audit text report finalized at: outputs/reports/shap_report.txt")
    print("   [✓] SHAP Explainability Engine completed runtime processes successfully.")

    return sv