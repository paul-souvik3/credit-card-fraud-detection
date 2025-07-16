import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("fraud_xgb_model.pkl")

st.set_page_config(page_title="💳 Credit Card Fraud Detection App", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This application uses a trained **XGBoost classifier** to detect the risk of credit card fraud based on transaction features.

### 🧾 Feature Input Guide
Please fill in the transaction details below. The model will predict whether the transaction is **fraudulent (1)** or **genuine (0)**.
""")

st.markdown("""
### 🔍 Feature Descriptions:

- **V14**: PCA component strongly correlated with fraudulent behavior. Enter a float (e.g., `-2.5`, `3.1`). Placeholder Example: `-2.3456`
- **V10**: Another PCA-derived feature linked to transaction patterns. Float expected (e.g., `-5.2`, `2.3`). Placeholder Example: `3.2178`
- **V12**: Related to cardholder behavior. Enter a float value (e.g., `-4.7`, `1.8`). Placeholder Example: `-4.1023`
- **V4**: Captures transaction frequency signals. Float required (e.g., `-0.5`, `3.7`). Placeholder Example: `1.2567`
- **V17**: Often linked to risk score by PCA. Float input (e.g., `-3.8`, `2.2`). Placeholder Example: `-1.7890`

📌 *Note: These are anonymized features derived from PCA, so original meanings are not disclosed due to privacy constraints.*
""")

# User Input
st.subheader("🔧 Enter Transaction Details")
v14 = st.number_input("V14", value=-2.3456, format="%.4f", help="Example: -2.3456")
v10 = st.number_input("V10", value=3.2178, format="%.4f", help="Example: 3.2178")
v12 = st.number_input("V12", value=-4.1023, format="%.4f", help="Example: -4.1023")
v4 = st.number_input("V4", value=1.2567, format="%.4f", help="Example: 1.2567")
v17 = st.number_input("V17", value=-1.7890, format="%.4f", help="Example: -1.7890")

# Predict Button
if st.button("🔍 Predict Fraud"):
    input_data = np.array([[v14, v10, v12, v4, v17]])
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 Alert: This transaction is predicted as **FRAUDULENT**.")
    else:
        st.success("✅ This transaction is predicted as **GENUINE**.")

    st.markdown(f"**🔢 Fraud Probability Score:** `{prediction_proba:.4f}`")

# Model Output Explanation
st.markdown("""
---
### 📌 Target Variable: `Class`
- `0` → Genuine Transaction
- `1` → Fraudulent Transaction

---
### ⚠️ Important Notice
This model is trained on a real-world **highly imbalanced dataset** with:
- 284,807 transactions
- Only ~492 fraud cases (~0.17%)

Despite best efforts (SMOTE balancing, tuning, outlier handling), **predictions might sometimes misclassify** due to:
- Imbalanced nature of data
- PCA-anonymized features
- Generalization limits

### 🔐 Use Case Restriction
This app is **not intended for production-level fraud prevention**. It is built for:
- Educational purposes ✅
- Prototyping & portfolio ✅
- Showcasing end-to-end ML pipeline ✅

---
### 📣 About This Project
- Built with: Python, XGBoost, Streamlit

""")



