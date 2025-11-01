# 🧠 Employee Attrition Prediction using Machine Learning

This repository contains the DSKC project by **Group 2**, focusing on the **Utilization of Machine Learning Techniques to Analyse IBM HR Dataset and Predict Employee Attrition**.

The primary objective is to develop a predictive model that accurately identifies employees at high risk of leaving a company. The project places special emphasis on handling the **class imbalance** inherent in attrition data, prioritizing **high recall** to ensure that at-risk employees are correctly flagged for proactive retention strategies.

---

## 👩‍💻 Contributors

* Nishkarsh Singhal
* Garima Singh
* Shaivee Sharma

---

## 📊 Data \& Preprocessing Pipeline

The project uses the publicly available **IBM Employee Attrition Dataset**, which includes **1,470 employee records and 35 features**.  
A key challenge is the **16:84 class imbalance**, with "Attrition = Yes" being the minority class.

**Data Processing Steps:**

1. **Initial Cleaning:** Dropped non-predictive columns (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`).
2. **Target Encoding:** Converted `Attrition` from 'Yes'/'No' to binary 1/0.
3. **Correlation Analysis:** Found high correlation between `JobLevel` and `MonthlyIncome`; removed `MonthlyIncome`.
4. **Encoding:**

   * **Tree-Based Models (RF, XGB, LGBM):** Used `LabelEncoder`.
   * **Linear/Neural Models (LR, MLP):** Used `pd.get\_dummies()` for One-Hot Encoding.

5. **Scaling:** Applied `StandardScaler` to gradient-based models (e.g., MLP, Logistic Regression).
6. **Imbalance Handling:** Used **SMOTE** (Synthetic Minority Oversampling Technique) *only on training data*.

---

## 🧪 Models Explored

Nine models were analyzed and benchmarked against the imbalanced dataset:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Gaussian Naive Bayes (GNB)
* Random Forest (RF)
* Extra Trees Classifier
* XGBoost
* LightGBM (LGBM)
* Multi-Layer Perceptron (MLP)

Models were fine-tuned and evaluated primarily on **Macro F1-Score** and **Recall** for the 'Attrition' class (not just accuracy).

---

## 🏆 Proposed Model: Stacking Ensemble

shortlisted both MLP and stacking ensemble model

Two models were evaluated: one optimized for overall F1 score, and another for recall. Since recall is more critical in attrition prediction (missing potential leavers is costlier than false alarms), the high-recall model was selected as the final version. 

**Base Learners (Level 0):**

* **Random Forest:** Reduces variance.
* **XGBoost:** Focuses on bias reduction.
* **LightGBM:** Fast, histogram-based boosting model.

**Meta-Learner (Level 1):**

* **Extra Trees Classifier:** Aggregates predictions from base learners efficiently.

---

## 📈 Results and Comparison with SOTA

Performance was benchmarked against the SOTA model from the 2025 paper:  
*“Developing a Hybrid Machine Learning Model for Employee Turnover Prediction: Integrating LightGBM and Genetic Algorithms.”*

### 🔧 Threshold Tuning

Default 0.5 threshold was suboptimal for imbalanced data.  
An optimal **threshold = 0.280** was chosen to maximize F1-score and Recall for the minority ('Attrition') class.

---

### 📊 Performance vs. SOTA

| Metric        | SOTA Model | Proposed Stacking Ensemble |
|----------------|:-----------:|:--------------------------:|
| \*\*F1 Score\*\*   | 0.73 | \*\*0.83\*\* |
| \*\*Precision\*\*  | 0.75 | \*\*0.88\*\* |
| \*\*Recall\*\*     | 0.72 | \*\*0.81\*\* |
| \*\*Accuracy\*\*   | 0.78 | \*\*0.80\*\* |

---

### 📉 Minority Class (Attrition = 1)

**Model:** `model\_ensemble\_high\_recall\_final.ipynb`  
**Optimal Threshold:** `0.280`

| Metric | Score |
|:--------|:------:|
| Accuracy | 79.93% |
| Precision | 0.43 |
| \*\*Recall\*\* | \*\*0.83\*\* |
| \*\*F1-Score\*\* | \*\*0.57\*\* |

> ✅ The model achieves high recall (0.83), meaning it correctly identifies 83% of employees likely to leave.


---

## 🌍 Generalizability

Validated on additional datasets:

1. **Watson Healthcare Dataset**
2. **Synthetic Employee Attrition Dataset (Stealth Technologies)**

Strong performance was maintained across datasets, confirming robustness.

---

## 💡 Conclusion

The **Stacking Ensemble Model** outperforms the SOTA benchmark and excels at identifying employees at risk of attrition.  
By optimizing for **recall**, the model minimizes false negatives — ensuring that HR teams can proactively retain key personnel.

> \*\*Key Achievement:\*\*  
> - Recall (Attrition = 1): \*\*0.83\*\*  
> - F1-Score: \*\*0.73\*\*  
> - Balanced trade-off between accuracy and sensitivity.

---

## 📁 Repository Structure

├── data

│   ├── stealth\_tech\_dataset

│   │   ├── test\_modified.csv

│   │   └── train\_modified.csv

│   ├── ATTRITION DATASET\_IBM.csv

│   └── watson\_healthcare\_modified.csv

├── results \\ research\_poster

│   └── DSKC Research Poster.pdf

├── src

│   ├── model\_ensemble\_high\_recall\_final.ipynb

│   └── model\_MLP\_high\_F1.ipynb

├── weekly\_reports

│   ├── DSKC Group-2 Week 1 Report.pdf

│   ├── DSKC Group-2 Week 2 Report.pdf

│   ├── DSKC Group-2 Week 3 Report.pdf

│   └── DSKC Group-2 Week 4 Report.pdf

└── readme.md

