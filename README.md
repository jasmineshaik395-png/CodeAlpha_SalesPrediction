# 📈 Sales Prediction using Python
### CodeAlpha Data Science Internship — Task 4

---

## 📌 Project Overview

This project predicts future **product sales** based on advertising spend across three platforms — **TV, Radio, and Newspaper** — using regression and ensemble machine learning models.

The goal is to help businesses make data-driven marketing decisions by understanding how advertising budget allocation impacts sales outcomes.

---

## 📂 Repository Structure

```
CodeAlpha_SalesPrediction/
│
├── Advertising.csv                  # Dataset (200 records)
├── sales_prediction.py              # Main Python script
├── sales_prediction_analysis.png    # Output visualisation (auto-generated)
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 📊 Dataset

| Column      | Description                        |
|-------------|------------------------------------|
| `TV`        | TV advertising spend ($K)          |
| `Radio`     | Radio advertising spend ($K)       |
| `Newspaper` | Newspaper advertising spend ($K)   |
| `Sales`     | Product sales ($K) — **target**    |

- **200 rows**, **no missing values**
- Source: Classic ISLR Advertising dataset

---

## 🛠️ Steps Performed

1. **Data Loading & Exploration** — shape, dtypes, statistics, null check  
2. **Data Cleaning & Transformation** — type casting, validation  
3. **Feature Engineering** — interaction term (TV×Radio), spend ratios, total spend  
4. **Feature Selection** — 7 features used for modelling  
5. **Model Training** — 4 regression models with 80/20 train-test split  
6. **Model Evaluation** — R², MAE, RMSE, 5-fold cross-validation  
7. **Visualisation** — 7-panel analysis chart  
8. **Business Insights** — actionable marketing recommendations  

---

## 🤖 Models Used

| Model               | R² Score | MAE   | RMSE  | CV R²  |
|---------------------|----------|-------|-------|--------|
| Linear Regression   | 0.988    | 0.448 | 0.616 | 0.990  |
| **Ridge Regression**| **0.988**| 0.445 | 0.610 | 0.990  |
| Random Forest       | 0.987    | 0.475 | 0.629 | 0.988  |
| Gradient Boosting   | 0.988    | 0.469 | 0.620 | 0.989  |

✅ **Best Model: Ridge Regression** (R² = 0.988)

---

## 💡 Key Business Insights

- **TV × Radio interaction** is the single strongest sales predictor
- **Radio** delivers the highest ROI per $1K spent (+0.20K sales)
- **Newspaper** has minimal impact — budget can be reallocated
- Increasing TV spend combined with Radio creates a **synergistic sales lift**

### Recommended Marketing Strategy
> Allocate the majority of budget to **TV** for volume, use **Radio** for high efficiency, and minimise **Newspaper** spend.

---

## 📉 Visualisations Generated

- TV / Radio / Newspaper vs Sales scatter plots with trend lines  
- Correlation heatmap  
- Model R² comparison bar chart  
- Actual vs Predicted sales plot  
- Random Forest feature importance  
- Advertising spend distribution  
- Residual plot  

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ShaikJasmine/CodeAlpha_SalesPrediction.git
cd CodeAlpha_SalesPrediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python sales_prediction.py
```

The script will print model metrics to the console and save **`sales_prediction_analysis.png`** to the current directory.

---

## 🔧 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🙏 Acknowledgements

- **CodeAlpha** for the internship opportunity  
- Dataset: ISLR Advertising dataset (James et al.)

---

## 📬 Connect

- 🔗 [LinkedIn](https://www.linkedin.com/in/ShaikJasmine) — *tag @CodeAlpha when you post!*
- 📧 [jasmineshaik395@gmail.com](mailto:jasmineshaik395@gmail.com)
- 💻 [GitHub](https://github.com/ShaikJasmine)
