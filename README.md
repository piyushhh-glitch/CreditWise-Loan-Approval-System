# 🏦 CreditWise — AI-Powered Loan Approval System

An end-to-end Machine Learning application that enables users to assess their loan eligibility based on personal, financial, credit, and employment factors. The platform delivers real-time loan approval predictions, applicant risk analysis, explainable decision insights, personalized improvement recommendations, and interactive data visualizations through a modern Streamlit dashboard.

🌐 **Live Demo:** https://creditwise-secure-bank-trust.streamlit.app/

---

## ✨ Features

- 🔍 Real-Time Loan Approval Prediction
- 📈 Applicant Risk Assessment Dashboard
- 💡 Explainable Rejection Reasons & Improvement Suggestions
- 📊 Interactive Data Analytics & Visualizations
- 🤖 Model Performance & Evaluation Report
- 📋 Comprehensive Financial Profile Analysis
- 🎨 Modern Banking-Themed Streamlit UI

---

## 🧠 Machine Learning Pipeline

```text
Dataset
   │
   ▼
Missing Value Imputation
   │
   ▼
Label Encoding + One-Hot Encoding
   │
   ▼
Feature Engineering
   │
   ▼
StandardScaler
   │
   ▼
Gaussian Naive Bayes
   │
   ▼
Loan Approval Prediction
```

### Feature Engineering

- DTI² (Debt-to-Income Ratio Squared)
- CreditScore² (Credit Score Squared)

### Model Configuration

- Gaussian Naive Bayes
- Train/Test Split: 80/20
- Random State: 42

---

## 📈 Model Performance

| Metric | Score |
|----------|----------|
| Accuracy | 86.5% |
| Precision | 78.3% |
| Recall | 77.7% |
| F1 Score | 77.7% |

---

## 🛠️ Tech Stack

CreditWise is built using a modern Machine Learning and Data Analytics stack for preprocessing, model training, visualization, and deployment.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
</p>

| Technology | Role |
|------------|------|
| 🐍 Python | Core programming language |
| 🎈 Streamlit | Interactive web application and deployment |
| 🐼 Pandas | Data preprocessing and analysis |
| 🔢 NumPy | Numerical computations |
| 🤖 Scikit-Learn | Machine learning pipeline and evaluation |
| 📊 Matplotlib | Data visualization and reporting |
| 📈 Seaborn | Exploratory data analysis |
| 🌐 Git & GitHub | Version control and project management |

---

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

---

## 🚀 Run Locally

```bash
git clone https://github.com/piyushhh-glitch/CreditWise-Loan-Approval-System.git

cd CreditWise-Loan-Approval-System

pip install -r requirements.txt

streamlit run app.py
```

Application will be available at:

```text
http://localhost:8501
```

---

## 📊 Dataset

The project uses a synthetic loan approval dataset containing:

- Applicant demographics
- Financial information
- Credit score and credit history
- Employment details
- Existing liabilities
- Loan characteristics
- Property information
- Savings and collateral details

**Total Records:** 1,000

---

## 👨‍💻 Author

**Piyush Thakur**

GitHub: https://github.com/piyushhh-glitch

---

⭐ If you found this project useful, consider giving it a star on GitHub!