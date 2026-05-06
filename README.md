# 💳 Financial Risk Trend Analysis using Python

A data analysis project exploring credit card fraud patterns using Python. Built to demonstrate skills in exploratory data analysis (EDA), risk identification, and data visualization — relevant to risk analytics and reporting roles in financial services.

---

## 📌 Project Overview

This project analyzes a real-world credit card transaction dataset to:
- Identify the proportion of fraudulent vs legitimate transactions
- Understand transaction amount patterns across fraud types
- Detect hourly risk trends to identify high-risk time windows
- Generate a structured summary report of key risk indicators

---

## 📊 Visualizations Generated

| Chart | Description |
|---|---|
| `chart1_fraud_vs_legit.png` | Bar chart comparing fraud vs legitimate transaction counts |
| `chart2_amount_distribution.png` | Histogram of transaction amounts by fraud class |
| `chart3_hourly_risk_trend.png` | Line chart showing fraud rate by hour of day |

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas** – data loading and manipulation
- **NumPy** – numerical operations
- **Matplotlib & Seaborn** – data visualization
- **Kaggle Dataset** – Credit Card Fraud Detection

---

## 📁 Project Structure

```
financial-risk-trend-analysis/
│
├── analysis.py              # Main analysis script
├── creditcard.csv           # Dataset (download from Kaggle)
├── outputs/
│   ├── chart1_fraud_vs_legit.png
│   ├── chart2_amount_distribution.png
│   ├── chart3_hourly_risk_trend.png
│   └── summary_report.txt
└── README.md
```

---

## 🚀 How to Run

**Step 1 – Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/financial-risk-trend-analysis.git
cd financial-risk-trend-analysis
```

**Step 2 – Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn
```

**Step 3 – Download the dataset**

Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
Download `creditcard.csv` and place it in the project root folder.

**Step 4 – Run the analysis**
```bash
python analysis.py
```

Charts and summary report will be saved in the `outputs/` folder.

---

## 📈 Key Findings

- The dataset is highly imbalanced — only ~0.17% of transactions are fraudulent
- Fraudulent transactions tend to have **lower average amounts** than legitimate ones
- Fraud risk shows a **distinct hourly pattern**, with a peak at a specific time window
- These insights can support **risk-based decision making and monitoring dashboards**

---

## 🎯 Relevance to Risk Analytics

This project simulates the kind of work done in **Risk Reporting & Analytics** roles:
- Working with large structured datasets
- Identifying trends and key risk indicators (KRIs)
- Generating visual reports for leadership review
- Supporting data-driven risk decisions

---

## 👩‍💻 Author

**Nishi Verma**  
