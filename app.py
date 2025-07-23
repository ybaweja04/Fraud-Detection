import streamlit as st
import pandas as pd
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# --- Model Loading ---
# Cached function to load models only once
@st.cache_resource
def load_models():
    """Load the pre-trained models and scaler from disk."""
    try:
        models = {
            "lr": joblib.load('models/logistic_regression_model.joblib'),
            "rf": joblib.load('models/random_forest_model.joblib'),
            "iso": joblib.load('models/isolation_forest_model.joblib'),
            "scaler": joblib.load('models/scaler.joblib')
        }
        # Store feature names from a trained model to ensure order
        models["feature_names"] = models["lr"].feature_names_in_
        return models
    except FileNotFoundError:
        st.error("🚨 Model files not found. Please ensure the 'models' directory is present and contains the .joblib files.")
        st.stop() # Stop the app if models can't be loaded

models = load_models()

# --- App Title and Description ---
st.title("💳 Real-Time Transaction Fraud Detection")
st.markdown("""
This interactive application uses machine learning to detect fraudulent credit card transactions.
- **How it works**: Adjust the transaction features in the sidebar on the left.
- **The Features**: The dataset contains features `V1` through `V28`, which are the result of a PCA transformation. This is done to protect sensitive user data. We are focusing on `V14`, a feature known to be highly indicative of fraud.
- **The Models**: We use three different models to provide a comprehensive analysis.
""")

st.divider()

# --- Sidebar for User Input ---
st.sidebar.header("⚙️ Adjust Transaction Features")
st.sidebar.info("Modify these values to simulate a transaction and see how the models react.")

# Default values for the features (can be median or mean of the original dataset)
default_values = { 'V1': -0.01, 'V2': 0.05, 'V3': 0.02, 'V4': 0.01, 'V5': -0.02, 'V6': 0.0, 'V7': -0.03, 'V8': 0.0, 'V9': -0.0, 'V10': -0.01, 'V11': 0.02, 'V12': -0.02, 'V13': 0.0, 'V14': -0.03, 'V15': 0.0, 'V16': 0.0, 'V17': -0.04, 'V18': 0.0, 'V19': 0.0, 'V20': 0.0, 'V21': 0.0, 'V22': 0.0, 'V23': 0.0, 'V24': 0.0, 'V25': 0.0, 'V26': 0.0, 'V27': 0.0, 'V28': 0.0 }


time_input = st.sidebar.slider(
    "Time Since First Transaction (seconds)", 0, 172792, 85000,
    help="Represents the time elapsed between this transaction and the very first one in the dataset."
)
amount_input = st.sidebar.number_input(
    "Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0,
    help="Enter the monetary value of the transaction."
)

st.sidebar.subheader("Key Fraud Indicator")
v14_input = st.sidebar.slider(
    "Anonymized Feature V14", -20.0, 10.0, default_values['V14'], step=0.1,
    help="V14 is one of the most important anonymized features for detecting fraud. Lower values are strongly correlated with fraudulent transactions."
)


# --- Prediction Logic and Display ---
if st.sidebar.button("Check for Fraud", use_container_width=True, type="primary"):
    # 1. Prepare input data
    input_data = default_values.copy()
    input_data.update({'Time': time_input, 'Amount': amount_input, 'V14': v14_input})

    df = pd.DataFrame([input_data])
    
    # Scale 'Amount' and 'Time' as they were during training
    df['scaled_amount'] = models['scaler'].transform(df[['Amount']])
    df['scaled_time'] = models['scaler'].transform(df[['Time']])
    
    # Drop original columns and ensure feature order is correct
    df = df.drop(['Time', 'Amount'], axis=1)
    df = df[models["feature_names"]]

    # 2. Get predictions from models
    lr_prob = models['lr'].predict_proba(df)[0][1]
    rf_prob = models['rf'].predict_proba(df)[0][1]
    iso_pred = models['iso'].predict(df)[0] # -1 for fraud, 1 for normal

    iso_result_text = "Fraud" if iso_pred == -1 else "Normal"
    
    # 3. Determine overall verdict based on a threshold
    FRAUD_THRESHOLD = 0.5  # 50% probability
    is_fraud = (rf_prob > FRAUD_THRESHOLD) or (iso_pred == -1) or (lr_prob > FRAUD_THRESHOLD)

    st.subheader("🤖 Model Predictions & Verdict")

    # Display the final verdict prominently
    if is_fraud:
        st.error("High Risk of Fraud Detected!", icon="🚨")
    else:
        st.success("Transaction Appears to be Normal", icon="✅")
    
    # Create tabs for detailed results and explanations
    tab1, tab2 = st.tabs(["📊 Prediction Summary", "🧠 Model Explanations"])

    with tab1:
        st.markdown("Here's a breakdown of each model's prediction:")
        col1, col2, col3 = st.columns(3)
        
        # Logistic Regression
        col1.metric(
            "Logistic Regression",
            f"{lr_prob:.2%}",
            "Fraud Probability"
        )

        # Random Forest
        col2.metric(
            "Random Forest",
            f"{rf_prob:.2%}",
            "Fraud Probability",
            delta_color="inverse" if rf_prob > FRAUD_THRESHOLD else "normal"
        )
        
        # Isolation Forest
        col3.metric(
            "Isolation Forest",
            iso_result_text,
            "Anomaly Detection",
            delta_color="inverse" if iso_pred == -1 else "normal"
        )

    with tab2:
        st.subheader("Understanding the Models")
        st.markdown("""
        - **Logistic Regression**: A straightforward and fast statistical model. It's a good baseline but may not capture complex patterns in the data. A higher percentage indicates a higher perceived probability of fraud.

        - **Random Forest**: A powerful and accurate model composed of many "decision trees." It's excellent at identifying complex, non-linear relationships in the data, making it very effective for fraud detection. We consider its output as a primary indicator.

        - **Isolation Forest**: An anomaly detection model. Instead of profiling normal transactions, it explicitly tries to "isolate" outliers. It's very effective for finding rare events, like fraud, that behave differently from the norm. A result of "Fraud" means the transaction is a significant outlier.
        """)

st.divider()
st.info(
    "**Disclaimer**: This is a demonstration tool. The predictions are based on a public dataset and should not be used for real financial decisions.",
    icon="ℹ️"
)