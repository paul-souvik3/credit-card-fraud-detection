# 💳 Credit Card Fraud Detection App

A Streamlit web application that uses a trained **XGBoost classification model** to detect the risk of credit card fraud based on anonymized transaction features.

---

## 🚀 Live App

👉 [Click here to access the live Streamlit app](https://your-streamlit-app-link-here)


---

## 📊 About the Dataset

* **Source**: Kaggle — Credit Card Fraud Detection
* **Samples**: 284,807 transactions
* **Fraud Cases**: 492 (\~0.17%) — heavily imbalanced
* **Features Used** (Top 5 selected based on importance):

  * `V14`: PCA component strongly correlated with fraud
  * `V10`: Linked to unusual transaction behavior
  * `V12`: Reflects customer spending behavior
  * `V4`: Captures transaction frequency signals
  * `V17`: Often tied to internal risk scores
* **Target Variable**: `Class`

  * `0`: Genuine Transaction
  * `1`: Fraudulent Transaction

---

## 🛠️ How It Works

Users input 5 anonymized PCA-derived features from a credit card transaction. The app:

* Loads a pre-trained `XGBoost` model (`fraud_xgb_model.pkl`)
* Predicts whether the transaction is **fraudulent** or **genuine**
* Displays the **fraud probability score**

---

## 🎯 App Features

* ✅ Clean Streamlit UI with input tooltips and examples
* ✅ Real-time fraud prediction with risk score
* ✅ Educates users on dataset and model behavior
* ✅ Clear disclaimers about limitations

---

## ⚠️ Model Disclaimer

* This model is trained on **anonymized and imbalanced** data.
* While tuned with SMOTE and hyperparameter optimization, **misclassifications may still occur**.
* Designed for:

  * Educational use ✅
  * Demonstrating end-to-end ML ✅
  * Prototyping and portfolios ✅
* ❌ Not suitable for production or real-time banking environments.

---

## 🧪 Example Inputs

| Feature | Placeholder Example |
| ------- | ------------------- |
| V14     | `-2.3456`           |
| V10     | `3.2178`            |
| V12     | `-4.1023`           |
| V4      | `1.2567`            |
| V17     | `-1.7890`           |

---

## 📝 Technologies Used

* Python
* Streamlit
* XGBoost
* scikit-learn
* pandas, numpy
* joblib

---

## 📁 Project Structure

```
Fraud_Detection/
├── app.py                  # Streamlit frontend app
├── fraud_xgb_model.pkl     # Trained ML model
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```
---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt







