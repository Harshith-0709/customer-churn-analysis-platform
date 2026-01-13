# 📉 Customer Churn Prediction Platform

An end-to-end Data Science project that predicts customer churn using machine learning and provides actionable business insights through data analysis, SQL, explainable AI, and a deployed web application.

🚀 **Live Demo:** https://customer-churn-analysis-platform-mbsqywkq3gpj28fwcbetmb.streamlit.app/

---

## 🔍 Problem Statement

Customer churn directly impacts company revenue and growth.  
The goal of this project is to:

- Predict customers who are likely to churn  
- Identify the key drivers of churn  
- Support data-driven retention strategies  
- Deploy a real-time prediction system for practical use  

---

## 🧠 Solution Overview

This project implements the complete Data Science lifecycle:

- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- SQL-Based Business Analysis  
- Feature Engineering  
- Machine Learning Modeling  
  - Logistic Regression  
  - Random Forest  
  - XGBoost (Final Model)  
- Model Evaluation (ROC-AUC, Precision, Recall, Confusion Matrix)  
- Explainable AI using SHAP  
- Real-Time Web Application using Streamlit  

---

## 🛠️ Tech Stack

- **Python:** Pandas, NumPy, Scikit-learn, XGBoost  
- **SQL:** SQLite  
- **Explainable AI:** SHAP  
- **Web App:** Streamlit  
- **Version Control:** Git & GitHub  
- **Deployment:** Streamlit Community Cloud  

---

## 📊 Dataset

**Telco Customer Churn Dataset (IBM Sample Dataset)**

Includes:

- Customer demographics  
- Services subscribed  
- Contract details  
- Payment methods  
- Monthly and total charges  
- Churn label  

---

## ⚙️ Machine Learning Models & Performance

| Model | ROC-AUC |
|-------|----------|
| Logistic Regression | ~0.84 |
| Random Forest | ~0.87 |
| XGBoost (Final Model) | ~0.89 |

---

## 🔍 Explainable AI (SHAP)

SHAP is used to:

- Identify the most important features influencing churn  
- Explain individual customer predictions  
- Improve transparency and business trust in the model  

---

## 🌐 Streamlit Web Application

### Features

- Upload customer CSV data  
- Predict churn probability for each customer  
- Binary churn classification (churn / not churn)  
- Download prediction results  
- Real-time inference using trained ML model  

🔗 **Live App:** https://customer-churn-analysis-platform-mbsqywkq3gpj28fwcbetmb.streamlit.app/

---

## 💡 Key Business Insights

- Month-to-month contract customers churn the most  
- Electronic check users have higher churn rates  
- Customers with tenure less than 12 months are high-risk  
- Higher monthly charges increase churn probability  
- Customers without tech support churn more frequently  
customer-churn-analysis-platform/
│
├── data/
├── notebooks/
├── sql/
├── app.py
├── model.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md

---

## 🧪 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
👤 Author

Harshith Valaboju
Aspiring Data Scientist | Machine Learning Enthusiast
---

## 📁 Project Structure

