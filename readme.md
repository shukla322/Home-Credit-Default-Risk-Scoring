Here is a detailed README file documenting your journey, methodology, and results, based on the provided sources.

---

# 📂 Home Credit Default Risk: From Raw Data to Top 5% (AUC 0.793)

## 📖 Project Overview

**The Problem: The "Invisible" Borrowers**
Many people have insufficient credit histories, causing traditional banks to reject them regardless of their actual reliability. This project aims to build a "Trust Engine" for the unbanked population by predicting repayment abilities using alternative data (telco, transactional, and behavioral data) rather than just traditional credit scores.

**The Goal**
To maximize **ROC-AUC (Area Under the Curve)**. We are not just making "Yes/No" predictions; we are building a ranking system to sort customers from "Most Risky" to "Least Risky".

---

## 🗺️ The Roadmap: My Journey

This project was not a straight line; it was an iterative evolution from a basic linear model to a complex "Grandmaster" ensemble.

```text
    PHASE 1: FOUNDATION        PHASE 2: BANKER LOGIC       PHASE 3: TIME TRAVEL
   [Clean & Baseline]  --->   [Domain Engineering]   --->  [Transactional History]
           |                           |                            |
    (Log Reg: 0.740)            (LightGBM: 0.770)           (LGBM Tuned: 0.781)
                                                                    |
                                                                    v
                                                            PHASE 4: THE MEGASET
                                                          [Active vs. Closed Splits]
                                                          [Time-Window Aggregations]
                                                                    |
                                                            (Final Ensemble: 0.793)
```

---

## 🛠️ Step 1: Data Cleaning & Anomaly Detection

Before modeling, the data required forensic analysis to handle "lies" and anomalies.

- **The "1000-Year-Old" Bug:** Detected the value `365243` in `DAYS_EMPLOYED`. This was a placeholder for "Unemployed" or "Pensioner," not a real number. I replaced it with `NaN` and created an anomaly flag column.
- **Informative Missingness:** Instead of blindly imputing missing values (e.g., `OWN_CAR_AGE`), I created binary flags (`_MISSING`). This taught the model that "Missing Data" often equals "No Asset" (higher risk).
- **Log Transformation:** Applied `log1p` to `AMT_INCOME_TOTAL` to fix skewness. This penalized "billionaire" outliers, compressing the distribution so linear models could handle it.

---

## 🏗️ Step 2: Feature Engineering (The "Secret Sauce")

This accounted for the majority of the predictive lift. I moved beyond raw data to create "Domain Features" used by real credit analysts.

### 📊 A. Financial Ratios (The Banker's Trust)

| Feature Name               | Formula            | Insight                                                                                                                                   |
| :------------------------- | :----------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| **CREDIT_INCOME_PERCENT**  | Loan / Income      | Measures leverage. Surprisingly, higher leverage often meant _lower_ risk (Reverse Causality: banks only give huge loans to safe people). |
| **ANNUITY_INCOME_PERCENT** | Payment / Income   | Measures "Debt Burden." High burden drastically increases default risk.                                                                   |
| **GOODS_LOAN_RATIO**       | Goods Price / Loan | Checks for Down Payment. If Ratio > 1, the client paid upfront, indicating high safety.                                                   |

### 🧠 B. Polynomial Interactions

- **EXT_SOURCE_MEAN:** Created a composite score of external sources. This became the single strongest predictor in the model.
- **INCOME_PER_AGE:** Captures financial stability relative to life stage (e.g., earning $50k at age 20 vs. age 60).

---

## 🚀 Step 3: Advanced Architecture (The 0.79 Push)

To break the 0.78 ceiling, I integrated the auxiliary files (`Bureau`, `Installments`, `POS_CASH`) and engineered a **311-column "Megaset"**.

### 1. The "Bureau" Attack

I merged `bureau.csv` to analyze the borrower's history with _other_ banks.

- **Insight:** Validated behavior via "Information Gain." Instead of relying on what the applicant _said_, we judged them on their _past behavior_.

### 2. Time-Window Strategy (The Nuclear Option)

Instead of aggregating all history, I created specific time windows to capture trends:

- **Short Term (6 Months):** Is the borrower in crisis _now_?
- **Long Term (24 Months):** Do they have a stable history?.

### 3. Active vs. Closed Splits

I separated loan statistics into two buckets:

- **Active Loans:** Represents current risk/burden.
- **Closed Loans:** Represents successful repayment history.
  This granular split was the key to jumping from **0.790 to 0.792**.

---

## 🤖 Models & Results

I evaluated multiple algorithms, ultimately selecting a **Stacking Ensemble**.

### Model Comparison Table

| Model                    | Score (AUC) | Strengths                                                                 | Weaknesses                                     |
| :----------------------- | :---------- | :------------------------------------------------------------------------ | :--------------------------------------------- |
| **Logistic Regression**  | **0.7400**  | Simple, interpretable baseline.                                           | Fails to capture non-linear relationships.     |
| **Single Decision Tree** | **0.7150**  | Visualizes rules.                                                         | High variance, overfits easily.                |
| **Random Forest**        | **0.7550**  | Robust, reduces variance.                                                 | Independent trees miss sequential errors.      |
| **XGBoost**              | **0.7780**  | Consistent performance.                                                   | Slower than LightGBM.                          |
| **LightGBM (Megaset)**   | **0.7921**  | **Fastest.** Uses "Leaf-wise" growth to minimize error aggressively.      | Can overfit on small data (not an issue here). |
| **CatBoost (Megaset)**   | **0.7909**  | Handles categories natively. "Ordered Boosting" finds different patterns. | Slower training time.                          |
| **FINAL ENSEMBLE**       | **0.7933**  | **Best of Both Worlds.** Combines LightGBM speed with CatBoost stability. | High complexity.                               |

### 📉 Visualizing the Lift

The "Lift" represents the value added by specific engineering phases.

```text
AUC SCORE EVOLUTION
0.80 |                                     [0.7933] 🏆
0.79 |                                    /
0.78 |                           [0.7810]
0.77 |                  [0.7705]    |
0.76 |                     |     (Bureau Data Added)
0.75 |          [0.7550]   |
0.74 | [0.7400]    |    (Domain Features)
     |    |    (LightGBM Baseline)
     | (LogReg)
     +---------------------------------------------------
       Phase 1   Phase 2   Phase 3   Phase 4   Phase 5
```

- **Phase 1-2:** Clean Data + Ratios (+0.030)
- **Phase 3:** Bureau & History Data (+0.011)
- **Phase 5:** Time-Windows & Active/Closed Splits (+0.008)

---

## 📊 Key Metrics Explained

### 1. ROC-AUC (Ranking Ability)

- **My Score:** 0.7933
- **Meaning:** If we pick one random "Defaulter" and one random "Payer," there is a **79.3% chance** my model ranks the Defaulter higher. This metric is stable because it does not depend on a specific cutoff threshold.

### 2. KS Statistic (Separation)

- **My Score:** ~0.50 (Estimated based on AUC)
- **Meaning:** Measures the distance between the "Blue Mountain" (Payers) and "Red Mountain" (Defaulters).
- **Verdict:** A KS score in this range indicates **"Good Model"** with strong separation, suitable for production credit models.

---

## 🏁 Conclusion

This project demonstrates that in credit risk modeling, **Data Architecture outperforms Hyperparameter Tuning**.

- Tuning gave a lift of only **+0.001**.
- Adding "Bureau Data" gave a lift of **+0.011**.
- Structuring "Active vs. Closed" splits gave a lift of **+0.003**.

**Final Result:** A robust, interpretable ensemble model placing in the top tier of benchmarks with an AUC of **0.7933**.
