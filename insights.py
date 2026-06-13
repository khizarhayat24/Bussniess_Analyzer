import pandas as pd
import numpy as np
import os

os.makedirs('outputs/reports', exist_ok=True)

def run_insights(df, results_df=None):
    print("\n" + "=" * 60)
    print("   BUSINESS INSIGHTS & REPORTING MODULE")
    print("=" * 60)

    df = df.copy()
    lines = []
    lines.append("=" * 60)
    lines.append("   TEYZIX CORE - BUSINESS INSIGHTS REPORT")
    lines.append("=" * 60)

    # ── 1. CUSTOMER SEGMENTATION (FIXED: Using Advanced Feature Engineering Segments)
    # Requirement 5: Based on behavior + spending patterns
    if 'customer_value' in df.columns:
        seg = df['customer_value'].value_counts()
        lines.append("\n[*] CUSTOMER SEGMENTATION BREAKDOWN:")
        for k, v in seg.items():
            pct = v / len(df) * 100
            lines.append(f"   {k}: {v} customers ({pct:.1f}%)")
        print("   ✔ Customer Segmentation stats processed.")
    else:
        print("   ⚠️ Warning: 'customer_value' column nahi mila! Feature Engineering check karo.")

    # ── 2. CHURN STATS (FIXED: Handles both Binary 1/0 and String 'Yes'/'No')
    if 'Churn' in df.columns:
        # Agar pichle step se 1/0 ho chuka hai to mean() hi churn rate hai, warna 'Yes' check karo
        if df['Churn'].dtype in [int, float, np.int64]:
            churn_rate = df['Churn'].mean() * 100
        else:
            churn_rate = df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
            
        lines.append(f"\n[*] OVERALL CHURN RATE: {churn_rate:.1f}%")
        print(f"   ✔ Overall Churn Rate: {churn_rate:.1f}%")

    # ── 3. TOP CHURN REASONS (FIXED: Binary alignment check)
    # Requirement 47: Top churn reasons
    # Agar column encoded hai to 1 check karo, warna 'Yes'
    is_churned = (df['Churn'] == 1) if df['Churn'].dtype in [int, np.int64] else (df['Churn'] == 'Yes')
    churned_df = df[is_churned]

    if len(churned_df) > 0:
        lines.append("\n[*] TOP CHURN REASONS & DRIVERS:")

        # Contract Type Analysis
        if 'Contract' in df.columns:
            top_contract = churned_df['Contract'].value_counts().idxmax()
            lines.append(f"   - Most Churned Contract Type : {top_contract}")

        # Payment Method Analysis
        if 'PaymentMethod' in df.columns:
            top_payment = churned_df['PaymentMethod'].value_counts().idxmax()
            lines.append(f"   - Most Churned Payment Method: {top_payment}")
            
        # Support Interaction Risk Analysis
        if 'support_interaction_count' in df.columns:
            avg_support_churn = churned_df['support_interaction_count'].mean()
            lines.append(f"   - Avg Support Interactions of Churned: {avg_support_churn:.1f} services")

        # Tenure Comparison
        if 'tenure' in df.columns:
            avg_tenure_churn = churned_df['tenure'].mean()
            avg_tenure_all   = df['tenure'].mean()
            lines.append(f"   - Avg Tenure of Churned Customers: {avg_tenure_churn:.1f} months")
            lines.append(f"   - Avg Tenure of Active Customers : {avg_tenure_all:.1f} months")

        print("   ✔ Top Churn drivers analyzed.")
    else:
        lines.append("\n[*] TOP CHURN REASONS: No churn data found or churn rate is 0%.")

    # ── 4. HIGH RISK CUSTOMER TRAITS (Requirement 48)
    # Yeh tab chalega jab risk prediction pipeline risk_category allocate karega (Low/Med/High)
    if 'risk_category' in df.columns:
        high_risk = df[df['risk_category'] == 'High Risk']
        lines.append(f"\n[*] HIGH RISK CUSTOMER SEGMENT PROFILE: {len(high_risk)} customers")

        numeric_cols = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges', 'support_interaction_count'] if c in df.columns]
        if len(high_risk) > 0 and numeric_cols:
            lines.append("   Avg profile characteristics of High-Risk customers:")
            for col in numeric_cols:
                avg = high_risk[col].mean()
                lines.append(f"      - Avg {col}: {avg:.2f}")
        print(f"   ✔ High Risk Profile Segment Analysed: {len(high_risk)} users.")
    else:
        lines.append("\n[*] HIGH RISK TRAITS: 'risk_category' column missing (Prediction step short hai).")

    # ── 5. REVENUE IMPACT ESTIMATION (Requirement 49)
    if len(churned_df) > 0 and 'MonthlyCharges' in df.columns:
        monthly_loss = churned_df['MonthlyCharges'].sum()
        annual_loss  = monthly_loss * 12
        lines.append(f"\n[*] REVENUE LOSS IMPACT ESTIMATION:")
        lines.append(f"   - Monthly Revenue At Risk: ${monthly_loss:,.2f}")
        lines.append(f"   - Annual Projected Loss  : ${annual_loss:,.2f}")
        print(f"   ✔ Financial Revenue Loss Impact computed successfully.")

    # ── 6. MODEL PERFORMANCE SUMMARY (Requirement 59)
    if results_df is not None:
        lines.append("\n[*] MACHINE LEARNING MODEL PERFORMANCE SUMMARY:")
        lines.append(results_df.to_string(index=False))
        
        # Best model finder via ROC-AUC score
        if 'ROC-AUC' in results_df.columns and 'Model' in results_df.columns:
            best = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
            lines.append(f"\n   >> Recommended Best Model for Deployment: {best}")

    # ── SAVE REPORT
    report_text = "\n".join(lines)
    with open('outputs/reports/business_insights.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Final combined dataset save
    df.to_csv('outputs/reports/final_customers_with_risk.csv', index=False)

    print("\n[✓] Reports and artifacts safely compiled:")
    print("    -> outputs/reports/business_insights.txt")
    print("    -> outputs/reports/final_customers_with_risk.csv")
    print("=" * 60 + "\n")

    return df