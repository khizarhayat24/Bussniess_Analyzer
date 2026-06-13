"""
ChurnIQ — Full Pipeline Runner
================================
Usage:
  python main.py           -> Full pipeline
  python main.py --shap    -> Pipeline + SHAP Explainability
  python main.py --all     -> Everything
"""

import argparse
import sys
import os
import numpy as np

from data_analysis       import run_data_analysis
from feature_engineering import run_feature_engineering
from visualization       import run_visualizations
from models              import run_models
from insights            import run_insights

def main(run_shap=False):
    print("\n" + "=" * 65)
    print("      ChurnIQ — CUSTOMER CHURN ANALYTICS PIPELINE")
    print("=" * 65)

    # STEP 1: DATA ANALYSIS
    print("\n[1/5] 📥 Data Analysis Module...")
    df = run_data_analysis(filepath='customerservices.csv')

    # STEP 2: FEATURE ENGINEERING
    print("\n[2/5] 🛠️ Feature Engineering & Segmentation...")
    df = run_feature_engineering(df)

    # STEP 3: VISUALIZATIONS
    print("\n[3/5] 📊 Data Visualizations...")
    run_visualizations(df)

    # STEP 4: ML MODELS — returns (df, results_df, champion_model, features)
    print("\n[4/5] 🤖 Machine Learning Models...")
    outputs = run_models(df)

    if isinstance(outputs, tuple) and len(outputs) == 4:
        df, results_df, champion_model, model_features = outputs
    elif isinstance(outputs, tuple) and len(outputs) == 2:
        df, results_df = outputs
        champion_model, model_features = None, None
    else:
        df = outputs
        results_df, champion_model, model_features = None, None, None

    # STEP 5: BUSINESS INSIGHTS
    print("\n[5/5] 📝 Business Insights Report...")
    df = run_insights(df, results_df)

    # BONUS: SHAP EXPLAINABILITY
    if run_shap:
        print("\n[BONUS] 🔮 SHAP Explainability...")
        try:
            from shap_explainability import run_shap_explainability

            if champion_model is not None and model_features is not None:
                run_shap_explainability(df=df, model=champion_model,
                                        features=model_features, model_name="Champion Model")
            else:
                # Fallback: retrain RF for SHAP
                from sklearn.ensemble import RandomForestClassifier
                critical_features = [
                    'tenure', 'MonthlyCharges', 'TotalCharges', 'usage_score',
                    'payment_risk', 'payment_stability', 'support_heavy',
                    'support_interaction_count', 'charges_ratio', 'avg_monthly_spend',
                    'spending_growth_index', 'Contract_encoded', 'PaymentMethod_encoded',
                    'InternetService_encoded'
                ]
                features = [f for f in critical_features if f in df.columns]
                X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
                y = df['Churn'].astype(int) if 'Churn' in df.columns else np.zeros(len(df))
                fallback_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
                fallback_rf.fit(X, y)
                run_shap_explainability(df=df, model=fallback_rf, features=features, model_name="Random Forest")

        except Exception as e:
            print(f"   ❌ SHAP Error: {e}")
            print("   -> Run: pip install shap")

    print("\n" + "=" * 65)
    print(" 🎉 PIPELINE COMPLETE!")
    print("=" * 65)
    print("  📁 Outputs:")
    print("  -> outputs/plots/          | 7+ Visualizations")
    print("  -> outputs/reports/        | Business Insights + Model Results")
    print("  -> outputs/models/         | Champion Model (champion_model.pkl)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChurnIQ Pipeline')
    parser.add_argument('--shap', action='store_true', help='Run SHAP explainability')
    parser.add_argument('--all',  action='store_true', help='Run everything')
    args = parser.parse_args()
    main(run_shap=args.shap or args.all)
