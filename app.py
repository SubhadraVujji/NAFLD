'''import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Define expected feature names
features = ['Age', 'ALT', 'BMI', 'DM.IFG', 'FBG', 'GGT', 'TG', 'AST.PLT']

# Load model and scaler
with open("fibrosis.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("A Machine learning based Combined Approach for Liver Fibrosis Diagnosis in NAFLD Using Biomarkers and Demographics")

# Collect user input for all 8 features
input_data = []
for feature in features:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    input_data.append(value)

# Predict button
if st.button("Predict"):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data], columns=features)
        
        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)
        
        # Display result
        st.success(f"Predicted Fibrosis Stage: {prediction[0]}")
    except Exception as e:
        st.error(f"Error: {e}")'''




import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64

# Function to set the background image
def set_background(image_file):
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{image_file}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Convert image to base64 format
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Call the function to set background
image_path = "background.png"  # Ensure this image is in your project directory
set_background(get_base64_image(image_path))

# Define expected feature names
features = ['Age', 'ALT', 'BMI', 'DM.IFG', 'FBG', 'GGT', 'TG', 'AST.PLT']

# Load model and scaler
with open("fibrosis.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("A Machine Learning Based Combined Approach for Liver Fibrosis Diagnosis in NAFLD")

st.markdown("### Enter patient details below:")

# Collect user input for all 8 features
input_data = []
for feature in features:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    input_data.append(value)

# Predict button
if st.button("Predict"):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data], columns=features)
        
        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)
        
        # Display result
        st.success(f"Predicted Fibrosis Stage: {prediction[0]}")
    except Exception as e:
        st.error(f"Error: {e}")

