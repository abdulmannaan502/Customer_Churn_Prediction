
# 📉 Customer Churn Prediction System

Predict which customers are likely to **churn (leave a service)** using machine learning.  
This project is end-to-end: from data analysis and modeling to explainability and deployment.

---

## 🧠 Problem Statement

Customer churn is a critical issue for subscription-based businesses.  
Acquiring new customers costs 5x more than retaining existing ones.

**Goal**: Predict customer churn ahead of time so companies can take preventive action.

---

## 📦 Dataset

- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Contains 7,000+ customer records with:
  - Demographics
  - Service subscriptions
  - Contract types
  - Monthly/total charges
  - Whether they churned or not

---

## 🛠️ Project Pipeline

### 1. 📊 **Exploratory Data Analysis**
- Identified key churn patterns (e.g. month-to-month contracts, short tenure)
- Handled missing values and converted `TotalCharges` to numeric

### 2. 🧼 **Data Preprocessing**
- One-hot encoded categorical features
- Scaled numeric values (e.g. tenure, charges)
- Train-test split with stratification

### 3. 🤖 **Model Training**
- Trained and evaluated:
  - Logistic Regression (best balance)
  - Random Forest
  - XGBoost (best recall)
- Evaluation Metrics:
  - Accuracy, Precision, Recall, F1 Score, AUC-ROC

### 4. 🧠 **Explainability with SHAP**
- SHAP summary plots to show global feature importance
- Force plots to explain individual predictions
- Key insight: Month-to-month contracts and low tenure increase churn risk

### 5. 🌐 **Streamlit Deployment**
- Interactive form to enter customer data
- Predicts churn + shows probability
- Explains **why** each prediction was made using top SHAP features

---

## 🚀 Streamlit App Preview

![Streamlit UI Screenshot](https://github.com/abdulmannaan502/Customer_Churn_Prediction/blob/main/Images/1.png)

Run locally:
```bash
cd app
streamlit run streamlit_app.py
```

---

## 🔍 Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 80.45%   | 64.95%    | 57.49% | 60.99%   | **83.60%** |
| Random Forest       | 78.96%   | 62.58%    | 51.87% | 56.73%   | 81.63% |
| XGBoost             | 77.83%   | 58.91%    | **54.81%** | 56.79%   | 81.97% |

---

## 💡 Key Insights

- Short tenure customers are more likely to churn
- Month-to-month contracts are highly correlated with churn
- Electronic check payment method = higher churn probability

---

## 🧰 Tech Stack

- Python, Pandas, NumPy
- Scikit-Learn, XGBoost, SHAP
- Matplotlib, Seaborn
- Streamlit (App Deployment)
- Jupyter Notebook

---

## 📁 Project Structure

```
customer_churn_prediction/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── model/
│   └── churn_xgboost_model.pkl
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

---

## 🙋‍♂️ Author

**Abdul Mannaan**  
Beginner ML Engineer working on end-to-end real-world projects.  
🔗 [Connect on LinkedIn](https://www.linkedin.com/in/abdulmannaan/)

---

## ⭐ Acknowledgements

- [Telco Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [SHAP for Explainable AI](https://github.com/slundberg/shap)
