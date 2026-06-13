import pandas as pd
import numpy as np
import os

# Output directory ensure karna
os.makedirs("outputs", exist_ok=True)

def run_data_analysis(filepath="customerservices.csv", output_filepath="outputs/clean_customers.csv"):
    """
    Teyzix Core Task 1: Data Analysis Module
    Loads, analyzes, handles missing values, and treats outliers cleanly.
    """
    print("\n" + "=" * 60)
    print("      DATA ANALYSIS MODULE (UPGRADED)")
    print("=" * 60)

    # 1. Load Dataset (Requirement 1.1)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Bhai, '{filepath}' file nahi mili! Path check kar.")
        
    df = pd.read_csv(filepath)

    # 2. Exploratory Data Analysis & Summary Stats (Requirement 1.2 & 1.4)
    print(f"\n[+] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n[+] Data Info:")
    df.info()

    print("\n[+] Summary Statistics (Numeric & Categorical):")
    print(df.describe(include="all"))

    print("\n[+] Missing Values Before Cleaning:")
    print(df.isnull().sum())

    # 3. Handle Specific Standard Columns (e.g., Telco formats)
    # TotalCharges kabhi kabhi spaces/string format mein hoti hai, convert to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 4. Robust Missing Value Imputation (Requirement 1.3)
    # Generic loop taake koi bhi column miss na ho
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                # Numeric ko median se fill karo
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                # Categorical/Object ko most frequent value (mode) se fill karo
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)

    # 5. Outlier Management via Capping/Winsorization (Requirement 1.3)
    # MonthlyCharges ya dusre continuous numeric columns ke liye safely handle karna
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        # ID aur Target (Churn) columns par outlier filter nahi lagana
        if "id" in col.lower() or "churn" in col.lower():
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Rows drop karne ke bajaye outliers ko cap (clip) karna safe hai
        # Taake row count change na ho aur valuable prediction data loss na ho
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    # 6. Save Clean Dataset
    df.to_csv(output_filepath, index=False)
    print(f"\n[✓] Cleaned dataset safely saved to: '{output_filepath}'")
    print(f"[✓] Final Cleaned Shape: {df.shape}")
    print("=" * 60 + "\n")

    return df

# Test run karne ke liye (agar file local script ke paas padi ho):
# if __name__ == "__main__":
#     df_clean = run_data_analysis()