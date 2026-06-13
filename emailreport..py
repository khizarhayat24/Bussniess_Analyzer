"""
Email Report Generator
----------------------
Automated churn analytics email report generator.
Reads latest weekly pipeline output and sends HTML email report.

Usage:
  python email_report.py                    -> Send report using customerservices.csv
  python email_report.py --data custom.csv  -> Use custom dataset
  python email_report.py --preview          -> Just save HTML, don't send email
  python email_report.py --to a@b.com c@d.com -> Override recipients
"""

import os
import argparse
import smtplib
import pandas as pd
import numpy as np
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ── EMAIL CONFIG ─────────────────────────────────────────────
# Set these via environment variables (recommended) or edit directly
CONFIG = {
    'smtp_host'    : os.getenv('SMTP_HOST', 'smtp.gmail.com'),
    'smtp_port'    : int(os.getenv('SMTP_PORT', '587')),
    'sender_email' : os.getenv('SENDER_EMAIL', 'your_email@gmail.com'),
    'sender_pass'  : os.getenv('SENDER_PASS',  'your_app_password'),
    'recipients'   : os.getenv('RECIPIENTS', 'recipient@example.com').split(','),
    'data_path'    : 'customerservices.csv',
    'output_dir'   : 'outputs/email_reports',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD & ANALYZE DATA
# ═══════════════════════════════════════════════════════════
def load_and_analyze(path):
    print(f"[*] Loading data from: {path}")
    df = pd.read_csv(path)

    # Clean
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Churn encode
    df['Churn_bin'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Features
    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['usage_score'] = (df['tenure'] * df['MonthlyCharges']) / 100

    # Quick RF churn probability
    from sklearn.ensemble import RandomForestClassifier
    feat_cols = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges'] if c in df.columns]
    X = df[feat_cols].fillna(0)
    y = df['Churn_bin']
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X, y)
    df['churn_prob'] = rf.predict_proba(X)[:, 1]
    df['risk_category'] = pd.cut(
        df['churn_prob'],
        bins=[-0.01, 0.30, 0.60, 1.00],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    print(f"   [OK] {len(df):,} records analyzed.")
    return df


# ═══════════════════════════════════════════════════════════
# STEP 2 — BUILD HTML EMAIL
# ═══════════════════════════════════════════════════════════
def build_html_report(df):
    run_date   = datetime.now().strftime('%d %B %Y, %I:%M %p')
    total      = len(df)
    churned    = int(df['Churn_bin'].sum())
    churn_rate = churned / total * 100
    high_risk  = int((df['risk_category'] == 'High Risk').sum())
    med_risk   = int((df['risk_category'] == 'Medium Risk').sum())
    low_risk   = int((df['risk_category'] == 'Low Risk').sum())
    rev_month  = float(df['MonthlyCharges'].sum())
    rev_risk   = float(df[df['risk_category'] == 'High Risk']['MonthlyCharges'].sum())
    rev_annual = rev_risk * 12
    avg_tenure_churned = df[df['Churn_bin'] == 1]['tenure'].mean() if churned > 0 else 0
    avg_charge_churned = df[df['Churn_bin'] == 1]['MonthlyCharges'].mean() if churned > 0 else 0

    # Top 5 high risk customers table rows
    display_cols = [c for c in ['customerID', 'tenure', 'MonthlyCharges', 'Contract', 'churn_prob'] if c in df.columns]
    top5 = df.sort_values('churn_prob', ascending=False).head(5)[display_cols]

    top5_rows = ""
    for _, row in top5.iterrows():
        prob = row.get('churn_prob', 0)
        badge_color = '#dc2626' if prob > 0.7 else '#f59e0b'
        top5_rows += f"""
        <tr>
            <td style="padding:10px 14px;border-bottom:1px solid #f3f4f6;font-weight:500;color:#111827">{row.get('customerID','N/A')}</td>
            <td style="padding:10px 14px;border-bottom:1px solid #f3f4f6;color:#6b7280">{int(row.get('tenure',0))} mo</td>
            <td style="padding:10px 14px;border-bottom:1px solid #f3f4f6;color:#6b7280">${row.get('MonthlyCharges',0):.0f}</td>
            <td style="padding:10px 14px;border-bottom:1px solid #f3f4f6;color:#6b7280">{row.get('Contract','N/A')}</td>
            <td style="padding:10px 14px;border-bottom:1px solid #f3f4f6">
                <span style="background:{badge_color};color:white;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600">
                    {prob:.0%}
                </span>
            </td>
        </tr>"""

    # Contract churn rates
    contract_rows = ""
    if 'Contract' in df.columns:
        for contract, grp in df.groupby('Contract'):
            rate = grp['Churn_bin'].mean() * 100
            bar_w = int(rate * 2.5)
            color = '#dc2626' if rate > 35 else '#f59e0b' if rate > 15 else '#10b981'
            contract_rows += f"""
            <tr>
                <td style="padding:8px 14px;color:#374151;font-weight:500">{contract}</td>
                <td style="padding:8px 14px">
                    <div style="background:#f3f4f6;border-radius:4px;height:10px;width:200px">
                        <div style="background:{color};height:10px;border-radius:4px;width:{min(bar_w,200)}px"></div>
                    </div>
                </td>
                <td style="padding:8px 14px;color:{color};font-weight:700">{rate:.1f}%</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ChurnIQ Weekly Report — {run_date}</title>
</head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:'Helvetica Neue',Arial,sans-serif;">

<!-- WRAPPER -->
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:32px 0">
<tr><td align="center">
<table width="640" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.08)">

  <!-- HEADER -->
  <tr>
    <td style="background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);padding:36px 40px">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td>
            <div style="font-size:22px;font-weight:700;color:#ffffff;letter-spacing:-0.5px">📊 ChurnIQ</div>
            <div style="font-size:13px;color:rgba(255,255,255,0.75);margin-top:4px">Customer Analytics Platform — Teyzix Core</div>
          </td>
          <td align="right">
            <div style="background:rgba(255,255,255,0.15);border-radius:8px;padding:8px 14px;display:inline-block">
              <div style="font-size:11px;color:rgba(255,255,255,0.8);font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Weekly Report</div>
              <div style="font-size:13px;color:#ffffff;font-weight:600;margin-top:2px">{run_date}</div>
            </div>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- ALERT BANNER (if high risk > 20%) -->
  {'<tr><td style="background:#fef2f2;border-left:4px solid #dc2626;padding:14px 40px"><span style="font-size:13px;color:#991b1b;font-weight:600">⚠️ High Alert: ' + str(high_risk) + ' customers (' + f"{high_risk/total*100:.1f}%" + ') are at high churn risk — immediate action recommended.</span></td></tr>' if high_risk/total > 0.10 else ''}

  <!-- KPI SECTION -->
  <tr>
    <td style="padding:32px 40px 0">
      <div style="font-size:11px;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:16px">Executive Summary</div>
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td width="25%" style="padding-right:10px">
            <div style="background:#f9fafb;border-radius:12px;padding:16px;border-top:3px solid #4f46e5;text-align:center">
              <div style="font-size:24px;font-weight:700;color:#111827">{total:,}</div>
              <div style="font-size:11px;color:#6b7280;margin-top:4px;font-weight:500">Total Customers</div>
            </div>
          </td>
          <td width="25%" style="padding-right:10px">
            <div style="background:#f9fafb;border-radius:12px;padding:16px;border-top:3px solid #dc2626;text-align:center">
              <div style="font-size:24px;font-weight:700;color:#dc2626">{churn_rate:.1f}%</div>
              <div style="font-size:11px;color:#6b7280;margin-top:4px;font-weight:500">Churn Rate</div>
            </div>
          </td>
          <td width="25%" style="padding-right:10px">
            <div style="background:#f9fafb;border-radius:12px;padding:16px;border-top:3px solid #f59e0b;text-align:center">
              <div style="font-size:24px;font-weight:700;color:#d97706">{high_risk:,}</div>
              <div style="font-size:11px;color:#6b7280;margin-top:4px;font-weight:500">High Risk</div>
            </div>
          </td>
          <td width="25%">
            <div style="background:#f9fafb;border-radius:12px;padding:16px;border-top:3px solid #10b981;text-align:center">
              <div style="font-size:24px;font-weight:700;color:#059669">${rev_month:,.0f}</div>
              <div style="font-size:11px;color:#6b7280;margin-top:4px;font-weight:500">Monthly Revenue</div>
            </div>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- REVENUE AT RISK -->
  <tr>
    <td style="padding:24px 40px 0">
      <div style="background:#fff7ed;border-radius:12px;padding:20px 24px;border:1px solid #fed7aa">
        <div style="font-size:13px;font-weight:700;color:#9a3412;margin-bottom:8px">💰 Revenue at Risk</div>
        <table width="100%" cellpadding="0" cellspacing="0">
          <tr>
            <td>
              <div style="font-size:28px;font-weight:700;color:#ea580c">${rev_risk:,.0f}<span style="font-size:14px;color:#9a3412;font-weight:500">/month</span></div>
              <div style="font-size:12px;color:#9a3412;margin-top:4px">Annualized: <strong>${rev_annual:,.0f}</strong> at risk from {high_risk:,} high-risk customers</div>
            </td>
            <td align="right">
              <div style="font-size:12px;color:#9a3412;text-align:right">
                Avg churned tenure: <strong>{avg_tenure_churned:.0f} months</strong><br>
                Avg churned charge: <strong>${avg_charge_churned:.0f}/mo</strong>
              </div>
            </td>
          </tr>
        </table>
      </div>
    </td>
  </tr>

  <!-- RISK DISTRIBUTION -->
  <tr>
    <td style="padding:24px 40px 0">
      <div style="font-size:11px;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:14px">Risk Distribution</div>
      <table width="100%" cellpadding="0" cellspacing="8">
        <tr>
          <td style="padding-bottom:10px">
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td width="90" style="font-size:12px;font-weight:600;color:#dc2626">🔴 High Risk</td>
                <td style="padding:0 12px">
                  <div style="background:#fee2e2;border-radius:6px;height:12px">
                    <div style="background:#dc2626;height:12px;border-radius:6px;width:{min(int(high_risk/total*300),300)}px"></div>
                  </div>
                </td>
                <td width="80" style="font-size:12px;color:#374151;text-align:right;font-weight:600">{high_risk:,} ({high_risk/total*100:.1f}%)</td>
              </tr>
            </table>
          </td>
        </tr>
        <tr>
          <td style="padding-bottom:10px">
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td width="90" style="font-size:12px;font-weight:600;color:#d97706">🟡 Medium</td>
                <td style="padding:0 12px">
                  <div style="background:#fef3c7;border-radius:6px;height:12px">
                    <div style="background:#f59e0b;height:12px;border-radius:6px;width:{min(int(med_risk/total*300),300)}px"></div>
                  </div>
                </td>
                <td width="80" style="font-size:12px;color:#374151;text-align:right;font-weight:600">{med_risk:,} ({med_risk/total*100:.1f}%)</td>
              </tr>
            </table>
          </td>
        </tr>
        <tr>
          <td>
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td width="90" style="font-size:12px;font-weight:600;color:#16a34a">🟢 Low Risk</td>
                <td style="padding:0 12px">
                  <div style="background:#dcfce7;border-radius:6px;height:12px">
                    <div style="background:#22c55e;height:12px;border-radius:6px;width:{min(int(low_risk/total*300),300)}px"></div>
                  </div>
                </td>
                <td width="80" style="font-size:12px;color:#374151;text-align:right;font-weight:600">{low_risk:,} ({low_risk/total*100:.1f}%)</td>
              </tr>
            </table>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- TOP 5 HIGH RISK CUSTOMERS -->
  <tr>
    <td style="padding:24px 40px 0">
      <div style="font-size:11px;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:14px">Top 5 Customers by Churn Probability</div>
      <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #f3f4f6;border-radius:10px;overflow:hidden">
        <tr style="background:#f9fafb">
          <th style="padding:10px 14px;text-align:left;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase">Customer ID</th>
          <th style="padding:10px 14px;text-align:left;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase">Tenure</th>
          <th style="padding:10px 14px;text-align:left;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase">Charge</th>
          <th style="padding:10px 14px;text-align:left;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase">Contract</th>
          <th style="padding:10px 14px;text-align:left;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase">Risk %</th>
        </tr>
        {top5_rows}
      </table>
    </td>
  </tr>

  <!-- CHURN BY CONTRACT -->
  <tr>
    <td style="padding:24px 40px 0">
      <div style="font-size:11px;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:14px">Churn Rate by Contract Type</div>
      <table width="100%" cellpadding="0" cellspacing="0">
        {contract_rows}
      </table>
    </td>
  </tr>

  <!-- RECOMMENDATIONS -->
  <tr>
    <td style="padding:24px 40px 0">
      <div style="font-size:11px;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:14px">Strategic Recommendations</div>
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr><td style="padding:10px 14px;background:#eff6ff;border-left:3px solid #3b82f6;border-radius:0 8px 8px 0;margin-bottom:8px;font-size:13px;color:#1e40af;display:block">
          📞 <strong>Immediate:</strong> Route {high_risk:,} high-risk customers to retention team for personalized outreach.
        </td></tr>
        <tr><td style="padding:4px 0"></td></tr>
        <tr><td style="padding:10px 14px;background:#eff6ff;border-left:3px solid #3b82f6;border-radius:0 8px 8px 0;font-size:13px;color:#1e40af;display:block">
          📋 <strong>Contract:</strong> Offer month-to-month customers a discounted annual plan upgrade to reduce churn risk.
        </td></tr>
        <tr><td style="padding:4px 0"></td></tr>
        <tr><td style="padding:10px 14px;background:#eff6ff;border-left:3px solid #3b82f6;border-radius:0 8px 8px 0;font-size:13px;color:#1e40af;display:block">
          💳 <strong>Payment:</strong> Migrate electronic check users to auto-pay — reduces churn risk significantly.
        </td></tr>
      </table>
    </td>
  </tr>

  <!-- FOOTER -->
  <tr>
    <td style="padding:32px 40px;margin-top:24px">
      <div style="border-top:1px solid #f3f4f6;padding-top:24px;text-align:center">
        <div style="font-size:12px;color:#9ca3af">
          Generated by <strong style="color:#4f46e5">ChurnIQ</strong> — Teyzix Core Internship · ML-INT-1<br>
          This is an automated report. Do not reply to this email.<br>
          <span style="font-size:11px">{run_date}</span>
        </div>
      </div>
    </td>
  </tr>

</table>
</td></tr>
</table>

</body>
</html>"""

    return html


# ═══════════════════════════════════════════════════════════
# STEP 3 — SAVE HTML LOCALLY
# ═══════════════════════════════════════════════════════════
def save_html(html):
    datestamp = datetime.now().strftime('%Y_%m_%d_%H%M')
    path = f"{CONFIG['output_dir']}/churniq_report_{datestamp}.html"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"   [✓] HTML report saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════
# STEP 4 — SEND EMAIL
# ═══════════════════════════════════════════════════════════
def send_email(html, html_path, recipients):
    subject = f"📊 ChurnIQ Weekly Churn Report — {datetime.now().strftime('%d %b %Y')}"

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = CONFIG['sender_email']
    msg['To']      = ', '.join(recipients)

    # Plain text fallback
    plain = f"""ChurnIQ Weekly Churn Report
Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}

Please view this email in an HTML-compatible client to see the full report.

This report was generated by ChurnIQ — Teyzix Core Internship ML-INT-1.
"""
    msg.attach(MIMEText(plain, 'plain'))
    msg.attach(MIMEText(html, 'html'))

    # Attach HTML file
    with open(html_path, 'rb') as f:
        attachment = MIMEBase('application', 'octet-stream')
        attachment.set_payload(f.read())
    encoders.encode_base64(attachment)
    attachment.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(html_path)}"')
    msg.attach(attachment)

    print(f"\n[*] Connecting to SMTP: {CONFIG['smtp_host']}:{CONFIG['smtp_port']}")
    print(f"[*] Sending to: {recipients}")

    with smtplib.SMTP(CONFIG['smtp_host'], CONFIG['smtp_port']) as server:
        server.ehlo()
        server.starttls()
        server.login(CONFIG['sender_email'], CONFIG['sender_pass'])
        server.sendmail(CONFIG['sender_email'], recipients, msg.as_string())

    print(f"   [✓] Email sent successfully to: {', '.join(recipients)}")


# ═══════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════
def run_email_report(data_path=None, recipients=None, preview_only=False):
    print("\n" + "=" * 60)
    print("   CHURNIQ — EMAIL REPORT GENERATOR")
    print("=" * 60)

    path = data_path or CONFIG['data_path']
    to   = recipients or CONFIG['recipients']

    # 1. Analyze
    df = load_and_analyze(path)

    # 2. Build HTML
    print("[*] Building HTML email report...")
    html = build_html_report(df)

    # 3. Save HTML
    html_path = save_html(html)

    # 4. Send or preview
    if preview_only:
        print(f"\n[PREVIEW MODE] Email NOT sent. Open the HTML file to preview:")
        print(f"   -> {html_path}")
    else:
        try:
            send_email(html, html_path, to)
        except smtplib.SMTPAuthenticationError:
            print("\n❌ SMTP Authentication Failed!")
            print("   Gmail users: use an App Password (not your main password)")
            print("   Guide: https://support.google.com/accounts/answer/185833")
            print(f"   HTML report saved at: {html_path}")
        except Exception as e:
            print(f"\n❌ Email send failed: {e}")
            print(f"   HTML report still saved at: {html_path}")

    print("=" * 60 + "\n")
    return html_path


# ═══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChurnIQ Email Report Generator')
    parser.add_argument('--data',    type=str,   default=None, help='Path to CSV dataset')
    parser.add_argument('--to',      nargs='+',  default=None, help='Recipient email addresses')
    parser.add_argument('--preview', action='store_true',      help='Save HTML only, do not send email')
    args = parser.parse_args()

    run_email_report(
        data_path    = args.data,
        recipients   = args.to,
        preview_only = args.preview
    )