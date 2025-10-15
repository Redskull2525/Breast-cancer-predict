import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Configuration ---
st.set_page_config(
    page_title="LR Model Cancer Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Feature names exactly as required by the model
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst',
    'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst',
    'fractal_dimension_worst'
]

# Default values for input widgets (based on common range in Breast Cancer dataset)
FEATURE_DEFAULTS = {
    # Mean features
    'radius_mean': (14.5, 6.9, 28.1, 0.1), # default, min, max, step
    'texture_mean': (19.2, 9.7, 39.3, 0.1),
    'perimeter_mean': (92.0, 43.7, 188.5, 0.1),
    'area_mean': (654.9, 143.5, 2501.0, 1.0),
    'smoothness_mean': (0.096, 0.052, 0.163, 0.001),
    'compactness_mean': (0.104, 0.019, 0.345, 0.001),
    'concavity_mean': (0.088, 0.0, 0.427, 0.001),
    'concave points_mean': (0.048, 0.0, 0.201, 0.001),
    'symmetry_mean': (0.181, 0.106, 0.304, 0.001),
    'fractal_dimension_mean': (0.062, 0.050, 0.097, 0.0001),

    # SE features (smaller ranges)
    'radius_se': (0.4, 0.1, 2.0, 0.001),
    'texture_se': (1.2, 0.3, 4.9, 0.001),
    'perimeter_se': (2.8, 0.8, 22.0, 0.1),
    'area_se': (40.0, 6.8, 542.2, 0.1),
    'smoothness_se': (0.007, 0.001, 0.031, 0.0001),
    'compactness_se': (0.025, 0.002, 0.135, 0.0001),
    'concavity_se': (0.031, 0.0, 0.396, 0.0001),
    'concave points_se': (0.011, 0.0, 0.053, 0.0001),
    'symmetry_se': (0.020, 0.008, 0.078, 0.0001),
    'fractal_dimension_se': (0.003, 0.001, 0.030, 0.0001),

    # Worst features
    'radius_worst': (17.5, 7.9, 36.0, 0.1),
    'texture_worst': (25.6, 12.0, 49.5, 0.1),
    'perimeter_worst': (115.0, 50.4, 251.2, 0.1),
    'area_worst': (880.0, 185.2, 4254.0, 1.0),
    'smoothness_worst': (0.132, 0.071, 0.223, 0.001),
    'compactness_worst': (0.254, 0.027, 1.058, 0.001),
    'concavity_worst': (0.272, 0.0, 1.252, 0.001),
    'concave points_worst': (0.114, 0.0, 0.291, 0.001),
    'symmetry_worst': (0.323, 0.156, 0.664, 0.001),
    'fractal_dimension_worst': (0.080, 0.055, 0.207, 0.0001),

    # ID feature
    'id': (1234567, 1, 99999999, 1),
}


# --- Model Loading and Prediction ---

@st.cache_resource # Use this decorator to load the model only once
def load_model():
    """Loads the pickled model from the file system."""
    try:
        with open('LR_model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("Model 'LR_model.pkl' loaded successfully.")
        return model
    except FileNotFoundError:
        st.error(f"Model file 'LR_model.pkl' not found in the application directory. Please ensure it is present.")
        # Return a string placeholder if the file is missing to prevent crashes
        return "MODEL_NOT_FOUND"

model = load_model()

# Removed predict_cancer_simulated as we are now aiming for real prediction
def predict_cancer_real(model_obj, input_data):
    """Performs real prediction using the loaded model."""
    
    if model_obj == "MODEL_NOT_FOUND":
        return 'B' # Default to benign if model is missing, but error is shown above

    # 1. Prepare input data as a DataFrame
    # Note: 'id' is included in input_data, but needs to be dropped for prediction
    df = pd.DataFrame([input_data])
    
    # 2. Extract features in the correct order for the model
    # The 'id' column is dropped as the model was likely trained without it
    features_only = df.drop(columns=['id']).reindex(columns=FEATURE_NAMES)
    
    # 3. Make the prediction
    try:
        prediction = model_obj.predict(features_only)
        # Assuming the model returns 'M' or 'B' directly, or 1/0 which we map
        result = 'M' if prediction[0] == 1 else 'B' # Adjust mapping (1->M, 0->B) if necessary
        # If your model returns 'M' or 'B' directly:
        # result = prediction[0] 
        return result
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return 'B' # Return default on error


# --- UI Functions ---

def format_name(name):
    """Formats feature names nicely (e.g., 'radius_mean' -> 'Radius Mean')"""
    return name.replace('_', ' ').title()

def create_input_widgets():
    """Creates the 32 input widgets in the sidebar and columns."""
    
    st.sidebar.title("Patient ID")
    # ID is separate and required
    id_default, id_min, id_max, id_step = FEATURE_DEFAULTS['id']
    patient_id = st.sidebar.number_input(
        "Patient ID (for tracking)",
        min_value=id_min,
        max_value=id_max,
        value=id_default,
        step=id_step
    )

    st.header("Tumor Measurements")
    st.markdown("Enter the 31 measurements to get a prognosis prediction.")
    
    # Create a dictionary to hold all 32 inputs
    input_data = {'id': patient_id}
    
    # Divide features into columns for a cleaner layout
    cols = st.columns(3) # Use 3 columns for desktop view
    
    for i, feature in enumerate(FEATURE_NAMES):
        col = cols[i % 3] # Cycle through the 3 columns
        
        default_val, min_val, max_val, step_val = FEATURE_DEFAULTS.get(feature, (1.0, 0.0, 10.0, 0.001))
        
        input_data[feature] = col.number_input(
            format_name(feature),
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=step_val,
            format="%f" if step_val < 0.1 else "%.1f",
            key=feature
        )
        
    return input_data

def display_prediction(prediction):
    """Displays the final prediction result."""
    
    st.divider()
    st.subheader("Prediction Result")
    
    if prediction == 'M':
        # Malignant (Have Cancer)
        st.error(
            "## Malignant (M)"
        )
        st.write("The model predicts a **malignant (M)** tumor. This suggests the presence of cancer.")
        st.warning("🚨 **Consult a Medical Professional:** This result is based on a machine learning model and requires confirmation from a qualified doctor.")
    else:
        # Benign (Not Have Cancer)
        st.success(
            "## Benign (B)"
        )
        st.write("The model predicts a **benign (B)** tumor. This suggests the mass is likely non-cancerous.")
        st.info("✅ **Consult a Medical Professional:** This result is based on a machine learning model and should not replace professional medical advice.")
        

# --- Main Application Logic ---

def main():
    st.title("Breast Cancer Prognosis Predictor 🔬")
    st.caption("Powered by Logistic Regression model (`LR_model.pkl`)")

    # Create input widgets and gather data
    input_data = create_input_widgets()
    
    st.sidebar.divider()
    
    # Prediction Button
    if st.sidebar.button("Predict Prognosis", type="primary"):
        # Predict using the real model (or placeholder if file missing)
        prediction = predict_cancer_real(model, input_data)
        
        # Display the result
        display_prediction(prediction)

if __name__ == "__main__":
    main()
