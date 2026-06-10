# 🏦 CreditWise — AI-Powered Loan Approval System

An intelligent Machine Learning application that helps assess loan applications using applicant demographics, financial profile, employment details, and credit history.

🌐 Live Demo: https://creditwise-secure-bank-trust.streamlit.app/

## ✨ Features

- 🔍 Real-time Loan Approval Prediction
- 📊 Interactive Data Insights Dashboard
- 🤖 Model Performance & Evaluation Report
- 📈 Applicant Risk Assessment Gauge
- 💡 Rejection Reasons & Improvement Suggestions
- 🎨 Professional Banking-Themed UI

## 🧠 Machine Learning Pipeline

Dataset → Missing Value Imputation → Encoding → Feature Engineering → StandardScaler → Gaussian Naive Bayes → Prediction

### Feature Engineering
- DTI² (Debt-to-Income Ratio Squared)
- CreditScore² (Credit Score Squared)

### Model
- Gaussian Naive Bayes
- Train/Test Split: 80/20
- Random State: 42

## 📈 Performance

| Metric | Score |
|----------|----------|
| Accuracy | 86.5% |
| Precision | 78.3% |
| Recall | 77.7% |
| F1 Score | 77.7% |

## 🛠️ Built With

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,github,git,vscode" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
</p>

## 📂 Project Structure

```text
CreditWise-Loan-Approval-System/
├── app.py
├── requirements.txt
├── README.md
├── dataset/
│   └── loan_approval_data.csv
└── notebook/
    └── creditwise.ipynb
```

## 🚀 Run Locally

```bash
git clone https://github.com/piyushhh-glitch/CreditWise-Loan-Approval-System.git

cd CreditWise-Loan-Approval-System

pip install -r requirements.txt

streamlit run app.py
```

## 👨‍💻 Author

**Piyush Thakur**

GitHub: https://github.com/piyushhh-glitch