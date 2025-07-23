# 💳 Real-Time Fraud Detection Dashboard

This project is a complete machine learning system that uses a Streamlit web interface to predict credit card fraud. It provides real-time predictions from three different models (Logistic Regression, Random Forest, and Isolation Forest) in a user-friendly and explanatory dashboard.

---

## ✨ Features

- **Multi-Model Analysis:** Compares predictions from three distinct models for a comprehensive risk assessment.
- **User-Friendly UI:** A clean and interactive dashboard built with Streamlit, featuring sidebar controls, tabs for organization, and clear verdict messages.
- **Helpful Explanations:** The app includes detailed explanations of each model's purpose and what their predictions mean.
- **Optimized Workflow:** Uses pre-trained models that are loaded once at startup for fast, real-time predictions on user input.

---

## 📂 Project Structure

```
├── models/                <- Folder for saved .joblib model files
├── app.py                 <- The main Streamlit application script
└── requirements.txt       <- Required Python packages
```

---

## 🚀 How to Run Locally

Follow these steps to set up and run the project on your own machine.

**1. Clone the repository:**

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate
# For Windows: .\venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download the dataset:**
- Download the Credit Card Fraud Detection dataset from [this Kaggle link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- Create a `data` folder in the project root.
- Place the downloaded `creditcard.csv` file inside the `data` folder.

**5. Run the Streamlit app:**
```bash
python -m streamlit run app.py
```
The application should now be open in your web browser!

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- Imbalanced-learn
