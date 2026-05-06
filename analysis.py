import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── aesthetics ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})
COLORS = {"fraud": "#E63946", "legit": "#457B9D", "accent": "#F4A261"}
os.makedirs("outputs", exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("creditcard.csv")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. Basic EDA ──────────────────────────────────────────────────────────────
total       = len(df)
fraud_count = df["Class"].sum()
legit_count = total - fraud_count
fraud_pct   = fraud_count / total * 100

print(f"\n📊 Dataset Overview")
print(f"  Total Transactions : {total:,}")
print(f"  Fraudulent         : {fraud_count:,}  ({fraud_pct:.2f}%)")
print(f"  Legitimate         : {legit_count:,}  ({100 - fraud_pct:.2f}%)")
print(f"\n  Amount Stats (all transactions):")
print(df["Amount"].describe().to_string())

# ── 3. Chart 1 – Fraud vs Legitimate Transactions ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ["Legitimate", "Fraudulent"],
    [legit_count, fraud_count],
    color=[COLORS["legit"], COLORS["fraud"]],
    edgecolor="white", linewidth=1.5, width=0.5
)
for bar, val in zip(bars, [legit_count, fraud_count]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_title("Transaction Distribution: Legitimate vs Fraudulent",
             fontsize=14, fontweight="bold", pad=14)
ax.set_ylabel("Number of Transactions", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, legit_count * 1.12)
plt.tight_layout()
plt.savefig("outputs/chart1_fraud_vs_legit.png")
plt.close()
print("\n✅ Chart 1 saved → outputs/chart1_fraud_vs_legit.png")

# ── 4. Chart 2 – Transaction Amount Distribution ─────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df[df["Class"] == 0]["Amount"].clip(upper=500),
        bins=60, color=COLORS["legit"], alpha=0.7, label="Legitimate", edgecolor="white")
ax.hist(df[df["Class"] == 1]["Amount"].clip(upper=500),
        bins=60, color=COLORS["fraud"], alpha=0.8, label="Fraudulent", edgecolor="white")
ax.set_title("Transaction Amount Distribution (clipped at $500)",
             fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Transaction Amount ($)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("outputs/chart2_amount_distribution.png")
plt.close()
print("✅ Chart 2 saved → outputs/chart2_amount_distribution.png")

# ── 5. Chart 3 – Hourly Risk Trend ───────────────────────────────────────────
df["Hour"] = (df["Time"] // 3600).astype(int) % 24
hourly = df.groupby("Hour")["Class"].agg(["sum", "count"])
hourly["fraud_rate"] = hourly["sum"] / hourly["count"] * 100

fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(hourly.index, hourly["fraud_rate"],
                alpha=0.25, color=COLORS["fraud"])
ax.plot(hourly.index, hourly["fraud_rate"],
        color=COLORS["fraud"], linewidth=2.5, marker="o", markersize=5)
peak_hour = hourly["fraud_rate"].idxmax()
ax.axvline(peak_hour, linestyle="--", color=COLORS["accent"],
           linewidth=1.5, label=f"Peak risk hour: {peak_hour}:00")
ax.set_title("Hourly Fraud Rate Trend", fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Hour of Day", fontsize=11)
ax.set_ylabel("Fraud Rate (%)", fontsize=11)
ax.set_xticks(range(0, 24))
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("outputs/chart3_hourly_risk_trend.png")
plt.close()
print("✅ Chart 3 saved → outputs/chart3_hourly_risk_trend.png")

# ── 6. Summary Report ─────────────────────────────────────────────────────────
avg_fraud_amt = df[df["Class"] == 1]["Amount"].mean()
avg_legit_amt = df[df["Class"] == 0]["Amount"].mean()

report = f"""
═══════════════════════════════════════════════════════
        FINANCIAL RISK TREND ANALYSIS – SUMMARY
═══════════════════════════════════════════════════════

Dataset        : Kaggle Credit Card Fraud Detection
Total Records  : {total:,}

── Transaction Breakdown ───────────────────────────────
  Legitimate   : {legit_count:,}  ({100 - fraud_pct:.2f}%)
  Fraudulent   : {fraud_count:,}   ({fraud_pct:.2f}%)

── Amount Analysis ─────────────────────────────────────
  Avg Legit Amount   : ${avg_legit_amt:.2f}
  Avg Fraud Amount   : ${avg_fraud_amt:.2f}

── Risk Trend ──────────────────────────────────────────
  Peak Fraud Hour    : {peak_hour}:00
  Max Hourly Rate    : {hourly['fraud_rate'].max():.2f}%

── Key Insights ────────────────────────────────────────
  • Dataset is highly imbalanced (~{fraud_pct:.1f}% fraud)
  • Fraud transactions tend to be lower in amount
  • Fraud risk peaks around hour {peak_hour}:00 of the day

═══════════════════════════════════════════════════════
"""
print(report)
with open("outputs/summary_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("✅ Summary report saved → outputs/summary_report.txt")
print("\n🎉 All done! Check the 'outputs/' folder.")