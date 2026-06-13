import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def run_feature_engineering(df):
    print("\n" + "=" * 60)
    print("   FEATURE ENGINEERING & SEGMENTATION MODULE")
    print("=" * 60)

    df = df.copy()

    # ── 1. TENURE GROUPS
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[-1, 12, 24, 48, 60, np.inf],
            labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5+yr']
        )
        print("   ✔ tenure_group created")

    # ── 2. USAGE SCORE
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['usage_score'] = (df['tenure'] * df['MonthlyCharges']) / 100
        print("   ✔ usage_score created")

    # ── 3. PAYMENT PATTERN FEATURES
    if 'PaymentMethod' in df.columns:
        df['payment_risk'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
        df['payment_stability'] = (df['PaymentMethod'].isin([
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])).astype(int)
        print("   ✔ payment_risk & payment_stability created")

    # ── 4. SUPPORT INTERACTION FEATURES
    support_cols = ['TechSupport', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection']
    existing_support_cols = [col for col in support_cols if col in df.columns]
    if len(existing_support_cols) > 0:
        df['support_interaction_count'] = df[existing_support_cols].eq('Yes').sum(axis=1)
        df['support_heavy'] = (df['support_interaction_count'] >= 2).astype(int)
        print("   ✔ support_interaction_count & support_heavy created")

    # ── 5. TREND / BEHAVIOR FEATURES
    if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns:
        df['charges_ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)
        print("   ✔ charges_ratio created")

    if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)
        df['spending_growth_index'] = df['MonthlyCharges'] / (df['avg_monthly_spend'] + 1)
        print("   ✔ avg_monthly_spend & spending_growth_index created")

    # ── 6. CUSTOMER VALUE SEGMENTATION
    if 'TotalCharges' in df.columns and 'usage_score' in df.columns:
        spend_score = pd.qcut(df['TotalCharges'], q=3, labels=[1, 2, 3], duplicates='drop').astype(int)
        usage_rank  = pd.qcut(df['usage_score'],  q=3, labels=[1, 2, 3], duplicates='drop').astype(int)
        final_score = spend_score + usage_rank
        df['customer_value'] = pd.cut(
            final_score, bins=[0, 3, 5, 6],
            labels=['Low Value', 'Medium Value', 'High Value']
        )
        print("   ✔ customer_value segmentation created")

    # ── 7. CRITICAL ENCODED COLUMNS FOR MODEL FEATURES
    # These specific columns must exist as _encoded for the model feature list
    critical_encode = ['Contract', 'PaymentMethod', 'InternetService']
    for col in critical_encode:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            print(f"   ✔ {col}_encoded created")

    # ── 8. ENCODE ALL REMAINING CATEGORICAL COLUMNS
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    exclude = ['customerID', 'Churn', 'Churn_encoded'] + [c + '_encoded' for c in critical_encode]

    saved_encoders = {}
    for col in cat_cols:
        if col not in exclude:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            saved_encoders[col] = le
            print(f"   ✔ {col} encoded")

    # ── 9. CHURN TARGET ENCODING
    if 'Churn' in df.columns:
        if df['Churn'].dtype == object:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
            print("   ✔ Churn encoded to binary (0/1)")

    # ── 10. DROP customerID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        print("   ✔ Dropped customerID")

    print(f"\n✅ Feature Engineering Done! Final Shape: {df.shape}")
    print("=" * 60 + "\n")
    return df
