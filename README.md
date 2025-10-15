
## 🧠 Breast Cancer Prediction App

This project uses **Machine Learning** to predict whether a breast tumor is **benign or malignant** based on key diagnostic features.
The trained model (`best_LR.pkl`) is deployed in a **Streamlit web application** that allows users to input patient data and receive instant predictions.

---

### 🚀 Project Overview

The app is powered by a **Logistic Regression** model trained on a breast cancer dataset.
It takes multiple numerical and categorical inputs (like mean radius, texture, area, smoothness, etc.) and predicts the **cancer class**:

* **0 → Benign (Non-cancerous)**
* **1 → Malignant (Cancerous)**

---

### 🧩 Features

✅ User-friendly web interface built with **Streamlit**
✅ Real-time prediction using a trained ML model (`best_LR.pkl`)
✅ Input validation for all features
✅ Error handling for missing model files
✅ Works locally and on Streamlit Cloud

---

### 🗂️ Project Structure

```
📁 breast-cancer-predict/
│
├── app.py                      # Streamlit application
├── best_LR.pkl                 # Trained Logistic Regression model
├── Breast-cancer-predict.ipynb # Jupyter notebook (training code)
└── requirements.txt            # Python dependencies
```

---

### ⚙️ Installation & Setup

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/breast-cancer-predict.git
cd breast-cancer-predict
```

#### 2️⃣ Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate       # On Windows
source venv/bin/activate    # On Mac/Linux
```

#### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

Streamlit app link - https://breast-cancer-predict-001100.streamlit.app/

### 🧮 How to Use

1. Enter all required input values (e.g., mean radius, texture, etc.).
2. Click **Predict**.
3. The app will display whether the tumor is **Benign** or **Malignant**.

---

### 🧰 Requirements

```
streamlit==1.39.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.4.2
```

---

### 📊 Model Information

* **Model Name:** best_LR.pkl
* **Algorithm:** Logistic Regression
* **Training Accuracy:** ~95% (based on dataset split)
* **Framework:** Scikit-learn
* **Input Format:** 20 feature columns (numeric)

---

### 💻 Deployment

You can easily deploy this project on:

* **Streamlit Cloud** → https://breast-cancer-predict-001100.streamlit.app
* **Render** → [https://render.com](https://render.com)
* **Hugging Face Spaces** → [https://huggingface.co/spaces](https://huggingface.co/spaces)

---

### 👨‍💻 Author

**Abhishek Shelke**
🎓 Master’s Student in Computer Science (SPPU, Pune)
💼 Aspiring Data Analyst & ML Developer
🌐 [GitHub Profile](https://github.com/redskull2525)

