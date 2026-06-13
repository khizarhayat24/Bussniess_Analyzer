import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt

os.makedirs('outputs/reports', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)

def run_models(df):
    print("\n" + "=" * 60)
    print("   MACHINE LEARNING MODELS MODULE")
    print("=" * 60)

    # ── 1. FEATURE SELECTION
    possible_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'usage_score',
        'payment_risk', 'payment_stability', 'support_heavy',
        'support_interaction_count', 'charges_ratio', 'avg_monthly_spend',
        'spending_growth_index', 'Contract_encoded', 'PaymentMethod_encoded',
        'InternetService_encoded'
    ]
    features = [f for f in possible_features if f in df.columns]
    print(f"\n🔹 Features for Training ({len(features)}): {features}")

    if 'Churn' not in df.columns:
        print("❌ Target column 'Churn' missing!")
        return df, None, None, features

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['Churn'].astype(int)
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # ── 2. TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"🔹 Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    X_all_sc   = scaler.transform(X)

    # Save scaler for weekly pipeline
    with open('outputs/models/standard_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    results = []
    models_dict = {}

    def evaluate(name, y_true, y_pred, y_prob):
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        auc  = roc_auc_score(y_true, y_prob)
        print(f"\nModel: {name}")
        print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        results.append({'Model': name, 'Accuracy': acc, 'Precision': prec,
                        'Recall': rec, 'F1': f1, 'ROC-AUC': auc})

    # ── 3. LOGISTIC REGRESSION
    print("\n🤖 Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced')
    lr.fit(X_train_sc, y_train)
    evaluate("Logistic Regression", y_test, lr.predict(X_test_sc), lr.predict_proba(X_test_sc)[:, 1])
    models_dict["Logistic Regression"] = (lr, True)

    # ── 4. RANDOM FOREST
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                                random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    evaluate("Random Forest", y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])
    models_dict["Random Forest"] = (rf, False)

    # ── 5. XGBOOST
    print("\n⚡ Training XGBoost...")
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        eval_metric='logloss', verbosity=0)
    xgb.fit(X_train, y_train)
    evaluate("XGBoost", y_test, xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1])
    models_dict["XGBoost"] = (xgb, False)

    # ── 6. RESULTS TABLE
    results_df = pd.DataFrame(results)
    print("\n📊 Model Comparison:")
    print(results_df.to_string(index=False))
    results_df.to_csv('outputs/reports/model_results.csv', index=False)

    # ── 7. CHAMPION MODEL SELECTION & SAVE
    best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
    print(f"\n🏆 Champion Model: {best_model_name}")
    champion_model, needs_scaling = models_dict[best_model_name]

    with open('outputs/models/champion_model.pkl', 'wb') as f:
        pickle.dump(champion_model, f)
    with open('outputs/models/champion_model_name.txt', 'w') as f:
        f.write(best_model_name)
    print("   ✔ Champion model saved to outputs/models/champion_model.pkl")

    # ── 8. FEATURE IMPORTANCE PLOT
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.plot(kind='bar', ax=ax, color='#2c3e50', edgecolor='black')
    ax.set_title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('outputs/plots/07_feature_importance.png', dpi=150)
    plt.close()
    print("   ✔ Feature importance chart saved.")

    # ── 9. CHURN PROBABILITY ON FULL DATASET
    df = df.copy()
    if needs_scaling:
        full_probs = champion_model.predict_proba(X_all_sc)[:, 1]
    else:
        full_probs = champion_model.predict_proba(X)[:, 1]

    df.loc[mask, 'churn_prob'] = full_probs
    df['risk_category'] = pd.cut(
        df['churn_prob'],
        bins=[-0.01, 0.30, 0.60, 1.00],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    print("   ✔ Churn probability and risk category assigned.")
    print("=" * 60 + "\n")

    # Return 4 values: df, results_df, champion_model, features
    return df, results_df, champion_model, features
