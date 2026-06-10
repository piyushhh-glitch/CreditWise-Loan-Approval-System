# 🏦 CreditWise — Loan Approval System

> Intelligent, unbiased ML-powered loan assessment for SecureTrust Bank.  
> Model: **Gaussian Naive Bayes** · Accuracy: 86.5% · Precision: 78.3%

## 📁 Project Structure

```
creditwise/
├── app.py                        ← Streamlit application (single file)
├── requirements.txt              ← Python dependencies
├── dataset/
│   └── loan_approval_data.csv    ← Training data (must be in this path)
└── README.md
```

## 🚀 Run Locally

```bash
# 1. Clone your repo
git clone https://github.com/piyushhh-glitch/CreditWise-Loan-Approval-System.git
cd CreditWise-Loan-Approval-System

# 2. Virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

Opens at **http://localhost:8501** 🎉

---

## ☁️ Deploy FREE on Streamlit Cloud

1. Commit `app.py`, `requirements.txt`, and `dataset/loan_approval_data.csv` to your GitHub repo
2. Go to **https://share.streamlit.io** → sign in with GitHub
3. Click **New app**
4. Set: Repository = your repo · Branch = `main` · Main file = `app.py`
5. Click **Deploy** — live in ~2 minutes, public URL generated automatically

---

## ✨ App Features

| Tab | What's Inside |
|---|---|
| **🔍 Loan Assessment** | Full applicant form → instant Approved/Rejected verdict · Rejection reasons + improvement tips · Feature impact chart · Live risk dashboard with gauge |
| **📊 Data Insights** | Approval split · Employment/property breakdowns · Credit score & DTI distributions · Boxplots by outcome |
| **🤖 Model Report** | Why NB was chosen · Accuracy/Precision/Recall/F1 · Confusion matrix with explanation · Full pipeline diagram |

## 🧠 ML Pipeline

```
Raw CSV → Impute (mean/mode) → LabelEncode → OneHotEncode
→ Feature Engineering (DTI², CreditScore²) → StandardScaler
→ GaussianNB → Predict + Explain
```
