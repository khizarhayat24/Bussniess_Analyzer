import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output plots directory architecture secure karna
os.makedirs('outputs/plots', exist_ok=True)
sns.set_theme(style='whitegrid')

def run_visualizations(df):
    print("\n" + "=" * 60)
    print("   DATA VISUALIZATION MODULE (UPGRADED)")
    print("=" * 60)
    
    df = df.copy()

    # ── 1. CHURN DISTRIBUTION (Requirement 3.3)
    if 'Churn' in df.columns:
        # Map back to strings strictly for presentation labels readability
        display_churn = df['Churn'].map({1: 'Churned (Yes)', 0: 'Active (No)'}).fillna(df['Churn'])
        churn_counts = display_churn.value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle('Customer Churn Distribution Profile', fontsize=14, fontweight='bold', color='#2c3e50')

        # Pie Chart
        churn_counts.plot(kind='pie', ax=axes[0], autopct='%1.1f%%', 
                          colors=['#2ecc71', '#e74c3c'], startangle=90, 
                          textprops={'fontweight': 'bold', 'fontsize': 11})
        axes[0].set_ylabel('')
        axes[0].set_title('Churn Proportional Share', fontsize=11, fontweight='bold')

        # Bar Chart
        sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=axes[1], palette=['#2ecc71', '#e74c3c'], edgecolor='black')
        axes[1].set_title('Churn Customer Count Headroom', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Total Volume of Customers')
        
        plt.tight_layout()
        plt.savefig('outputs/plots/01_churn_distribution.png', dpi=150)
        plt.close()
        print("   ✔ Plot 1: Churn Distribution Saved.")

    # ── 2. REVENUE TRENDS LINE PATTERN (Requirement 3.1)
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        trend = df.groupby('tenure')['MonthlyCharges'].mean()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(trend.index, trend.values, color='#2980b9', linewidth=2.5, marker='o', markersize=4, label='Avg Spend')
        ax.fill_between(trend.index, trend.values, alpha=0.15, color='#2980b9')
        
        ax.set_title('Customer Billing Evolution Trend Across Tenure Lifespan', fontsize=13, fontweight='bold', color='#2c3e50')
        ax.set_xlabel('Tenure Cohort (Duration in Months)')
        ax.set_ylabel('Mean Monthly Revenue Charges ($)')
        
        plt.tight_layout()
        plt.savefig('outputs/plots/02_revenue_trend.png', dpi=150)
        plt.close()
        print("   ✔ Plot 2: Billing Revenue Trend Curve Saved.")

    # ── 3. CORRELATION HEATMAP (FIXED: Overlap Matrix Bug) (Requirement 3.4)
    # Sirf top critical behavior features choose karte hain taake chart overlap na kare aur readable rahe
    critical_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'usage_score', 
        'support_interaction_count', 'charges_ratio', 'payment_risk', 'Churn'
    ]
    selectable = [col for col in critical_features if col in df.columns]
    
    if len(selectable) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df[selectable].corr()
        
        sns.heatmap(corr_matrix, ax=ax, annot=True, fmt='.2f', cmap='coolwarm', 
                    linewidths=0.75, square=True, annot_kws={'weight': 'bold', 'size': 10})
        
        ax.set_title('Behavioral Attributes Correlation Heatmap Matrix', fontsize=13, fontweight='bold', color='#2c3e50')
        plt.tight_layout()
        plt.savefig('outputs/plots/03_correlation_heatmap.png', dpi=150)
        plt.close()
        print("   ✔ Plot 3: Core Correlation Heatmap Matrix Saved.")

    # ── 4. CHURN BY CONTRACT CATEGORY (FIXED: KeyError Bug) (Requirement 3.5)
    # Check text mappings format
    contract_col = 'Contract' if 'Contract' in df.columns else ('Contract_encoded' if 'Contract_encoded' in df.columns else None)
    
    if contract_col and 'Churn' in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        
        # Aggregate logic mapping
        churn_by_contract = df.groupby(contract_col)['Churn'].mean().sort_values(ascending=False)
        
        sns.barplot(x=churn_by_contract.index, y=churn_by_contract.values, ax=ax, palette='Oranges_r', edgecolor='black')
        ax.set_title('Churn Risk Ratio Metrics across Contract Modalities', fontsize=13, fontweight='bold', color='#2c3e50')
        ax.set_xlabel('Contract Structure Classification')
        ax.set_ylabel('Statistical Churn Probabilities Rate')
        
        # Annotate labels calculation logic
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1%}', 
                        (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#d35400')
            
        plt.tight_layout()
        plt.savefig('outputs/plots/04_churn_by_contract.png', dpi=150)
        plt.close()
        print("   ✔ Plot 4: Contract Category Churn Rates Saved.")

    # ── 5. MONTHLY CHARGES DISTRIBUTION & BOXPLOT (FIXED: Boxplot Title Layer Bug)
    if 'MonthlyCharges' in df.columns and 'Churn' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle('Financial Monthly Charges Profile Behavior Analysis', fontsize=13, fontweight='bold', color='#2c3e50')

        # Histogram Distribution Plot
        sns.histplot(df['MonthlyCharges'], bins=30, kde=True, ax=axes[0], color='#9b59b6', edgecolor='black')
        axes[0].set_title('Global Monthly Billing Density Distribution', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Monthly Plan Amount Bin ($)')

        # Fixed Seaborn Boxplot configuration
        display_box_churn = df['Churn'].map({1: 'Churned', 0: 'Active'}).fillna(df['Churn'])
        sns.boxplot(x=display_box_churn, y=df['MonthlyCharges'], ax=axes[1], palette='Purples')
        axes[1].set_title('Plan Value Dispersion by Account Lifecycle Status', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Account Status')
        axes[1].set_ylabel('Charges Distribution ($)')

        plt.tight_layout()
        plt.savefig('outputs/plots/05_monthly_charges.png', dpi=150)
        plt.close()
        print("   ✔ Plot 5: Plan Financial Distribution Metrics Saved.")

    # ── 6. CUSTOMER SEGMENTATION DISTRIBUTION PIE CHART (Requirement 3.2)
    if 'customer_value' in df.columns:
        fig, ax = plt.subplots(figsize=(7, 7))
        segment_data = df['customer_value'].value_counts()
        
        segment_data.plot(kind='pie', ax=ax, autopct='%1.1f%%',
                          colors=['#f1c40f', '#bdc3c7', '#2ecc71'], startangle=140,
                          textprops={'fontweight': 'bold', 'fontsize': 11})
        
        ax.set_title('Algorithmic Value-Based Customer Segmentation Share', fontsize=13, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('')
        
        plt.tight_layout()
        plt.savefig('outputs/plots/06_customer_segmentation.png', dpi=150)
        plt.close()
        print("   ✔ Plot 6: Customer Profile Segmentation Share Saved.")

    print("\n[✓] All high-fidelity evaluation graphics compiled safely in 'outputs/plots/'")
    print("=" * 60 + "\n")
    return df