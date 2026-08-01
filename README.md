<div align="center">

# 🛡️ ChurnGuard

### Enterprise-Grade Customer Churn Prediction Platform

**Predict customer churn risk using Machine Learning and deliver actionable retention insights.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-006ACC?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

🚀 **Live Demo:** https://churnguard-ml.streamlit.app

<br>

⭐ If you like this project, consider giving it a star!

</div>

---

# 📖 Overview

**ChurnGuard** is an end-to-end Machine Learning application designed to predict customer churn in the telecommunications industry.

The platform analyzes customer demographics, subscribed services, contract information, and billing behavior to identify customers who are likely to leave.

The application combines:

- Machine Learning prediction engine
- Risk scoring system
- Feature importance analysis
- Business retention recommendations
- Modern SaaS-style Streamlit interface


## 💡 Business Problem

Customer acquisition is significantly more expensive than customer retention.

ChurnGuard helps businesses:

✅ Identify high-risk customers early  
✅ Understand churn-driving factors  
✅ Create targeted retention strategies  
✅ Protect recurring revenue  


---

# ✨ Features

| Feature | Description |
|---|---|
| 🎯 Real-Time Prediction | Instant churn probability prediction |
| 📊 Risk Classification | Low, Medium, High and Critical risk levels |
| 🧠 ML-Based Analysis | XGBoost powered prediction engine |
| 📈 Feature Importance | Understand factors influencing churn |
| 💼 Business Recommendations | Suggested retention actions |
| ⚡ Fast Inference | Cached model loading for quick predictions |
| 🎨 Premium Dashboard | Enterprise SaaS inspired UI |


---

# 🖥️ Application Preview

Add screenshots here:

```
docs/
 ├── dashboard.png
 └── prediction.png
```

Example:

![Dashboard](docs/dashboard.png)

![Prediction](docs/prediction.png)


---

# 🧠 Machine Learning Pipeline


```
Raw Customer Data
        |
        ↓
Data Cleaning
        |
        ↓
Feature Engineering
        |
        ↓
Binary Encoding + One Hot Encoding
        |
        ↓
Standard Scaling
        |
        ↓
XGBoost Model Training
        |
        ↓
Hyperparameter Optimization
        |
        ↓
Model Evaluation
        |
        ↓
Joblib Serialization
        |
        ↓
Streamlit Deployment
```


---

# 📊 Model Performance


| Metric | Score |
|---|---:|
| Accuracy | 81.4% |
| ROC-AUC | 0.856 |
| Precision | 0.78 |
| Recall | 0.72 |
| F1 Score | 0.75 |


## Model Comparison


| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 79.2% | 0.831 |
| Random Forest | 80.1% | 0.842 |
| **XGBoost (Tuned)** | **81.4%** | **0.856** |


---

# 📂 Dataset

Dataset:

**IBM Telco Customer Churn Dataset**

Source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn


Details:

- Records: 7,043 customers
- Original Features: 21
- Final Features: 23
- Target: Customer Churn
- Churn Rate: 26.5%


## Feature Categories


### 👤 Customer Information
- Gender
- Senior Citizen
- Partner
- Dependents


### 📡 Services
- Phone Service
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Streaming Services


### 💳 Account Information
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
- Tenure


---

# 🛠️ Tech Stack


## Programming

- Python 3.10+

## Machine Learning

- XGBoost
- Scikit-Learn
- Pandas
- NumPy

## Application

- Streamlit

## Deployment

- Streamlit Cloud
- GitHub
- Joblib


---

# 🚀 Installation


## Clone Repository


```bash
git clone https://github.com/mahadi-ur-rehman-pixel/customer-churn-prediction.git

cd customer-churn-prediction
```


## Create Virtual Environment


```bash
python -m venv venv
```


Activate:


Linux/Mac:

```bash
source venv/bin/activate
```


Windows:

```bash
venv\Scripts\activate
```


## Install Dependencies


```bash
pip install -r requirements.txt
```


## Run Application


```bash
streamlit run app.py
```


Open:


```
http://localhost:8501
```


---

# 📁 Project Structure


```
customer-churn-prediction/

│
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── docs/
│   ├── dashboard.png
│   └── prediction.png
│
└── .streamlit/
    └── config.toml
```


---

# 🔬 Model Details


## Preprocessing


### Binary Encoding

```
Yes → 1
No → 0

Male → 1
Female → 0
```


### One Hot Encoding

Converted:

```
Internet Service
Contract
Payment Method
```


### Feature Scaling

Applied StandardScaler on:

```
Tenure Months
Monthly Charges
Total Charges
```


---

# 💼 Business Applications


| Use Case | Value |
|---|---|
| Customer Retention | Identify customers before churn |
| Marketing Strategy | Personalized campaigns |
| Revenue Protection | Reduce customer loss |
| Customer Analytics | Understand behavior patterns |
| Executive Reporting | Churn monitoring dashboard |


---

# ⚠️ Limitations


- Model trained on telecom industry data only
- External market factors are not included
- Predictions represent probability, not certainty
- Historical patterns may change over time
- ML output should support business decisions, not replace them


---

# 🚀 Future Improvements


Possible upgrades:


- SHAP Explainable AI integration
- Batch CSV prediction system
- Customer history database
- Automated retention workflow
- AI chatbot for customer insights
- Advanced analytics dashboard


---

# 👨‍💻 Developer


**Mahadi Ur Rehman**

GitHub:

https://github.com/mahadi-ur-rehman-pixel


---

# 📜 License


This project is licensed under the MIT License.


---

<div align="center">

Built with ❤️ using Python, Machine Learning and Streamlit

⭐ Star this repository if you found it useful

</div># customer-churn-prediction
