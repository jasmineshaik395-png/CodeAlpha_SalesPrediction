# 📊 Unemployment Analysis with Python

> **Course:** Data Engineering | **Author:** [Your Name]
> **Domain:** Economics / Social Data | **Tools:** Python · Pandas · Matplotlib · Seaborn

---

## 📌 Project Overview

This project analyzes **India's unemployment data** from CMIE (Centre for Monitoring Indian Economy). It covers state-wise trends, rural vs urban comparison, Covid-19 impact analysis, and seasonal patterns using real-world data.

---

## 🗂️ Project Structure

```
CodeAlpha_UnemploymentAnalysis/
├── data/
│   ├── Unemployment_in_India.csv           # State-wise monthly data (2019–2020)
│   └── Unemployment_Rate_upto_11_2020.csv  # National + zone data with coordinates
├── notebooks/
│   └── Unemployment_Analysis.ipynb
├── outputs/
│   ├── unemployment_dashboard.png   # Full 7-panel analysis dashboard
│   ├── covid_impact.png             # Covid-19 impact comparison
│   ├── state_heatmap.png            # State × Month heatmap
│   ├── correlation_heatmap.png      # Feature correlations
│   ├── top_bottom_states.png        # Best & worst performing states
│   └── unemployment_report.txt      # Full statistical report
├── unemployment_analysis.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

### Dataset 1 — Unemployment in India
| Column | Description |
|---|---|
| Region | Indian state (28 states) |
| Date | Monthly date |
| Estimated Unemployment Rate (%) | State unemployment rate |
| Estimated Employed | Number of employed people |
| Estimated Labour Participation Rate (%) | Labour participation |
| Area | Rural / Urban |

### Dataset 2 — Unemployment Rate upto Nov 2020
| Column | Description |
|---|---|
| Region | Indian state |
| Date | Monthly date (Jan–Nov 2020) |
| Unemployment Rate (%) | Monthly rate |
| Region.1 | Zone (North/South/East/West/Northeast) |
| longitude/latitude | Geographic coordinates |

---

## 🦠 Covid-19 Impact

| Period | Avg Unemployment Rate |
|---|---|
| Pre-Covid (Jan–Mar 2020) | 9.23% |
| During Covid (Apr–Nov 2020) | 12.96% |
| **Spike** | **+3.73 percentage points** |
| **Peak** | **April 2020 → 75.85%** |

---

## 📈 Key Findings

| # | Insight |
|---|---|
| 1 | Covid-19 lockdown caused unemployment to spike by **+3.73%** points |
| 2 | Peak unemployment hit **75.85%** in April 2020 during national lockdown |
| 3 | **Urban areas** have higher unemployment than Rural areas |
| 4 | **Northeast zone** shows the highest average unemployment |
| 5 | Employment gradually recovered after **June 2020** |
| 6 | **Haryana & Tripura** consistently rank among highest unemployment states |
| 7 | Labour participation rate declined sharply during lockdown months |
| 8 | States with higher labour participation show lower unemployment rates |

---

## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_UnemploymentAnalysis.git
cd CodeAlpha_UnemploymentAnalysis
pip install -r requirements.txt
python unemployment_analysis.py
```

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| **Pandas** | Data loading, cleaning, grouping |
| **NumPy** | Numerical operations |
| **Matplotlib** | Dashboards, trend charts |
| **Seaborn** | Heatmaps, boxplots, KDE |
| **Jupyter Notebook** | Interactive exploration |

---

*Submitted as part of the CodeAlpha Data Science Internship — Task 1 — Unemployment Analysis Project.*
