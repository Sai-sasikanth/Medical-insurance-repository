import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import BytesIO

# Set page config with medical-themed colors
st.set_page_config(page_title="Medical Price Predictor", layout="centered")

# Custom CSS for background and text styling
st.markdown("""
    <style>
   .stApp {
        background-color: #3385ff;
        color: #004d4d;
    }

    .main {
        background-color: 	#ffffff;
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #007c91;
    }
    .stButton>button {
        background-color: #007c91;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stSlider>div {
        margin-top: 10px;
    }
    .input-field {
        margin-top: 20px;
        margin-bottom: 20px;
        color:000000;
    }
    .stSelectbox>div, .stSlider>div {
        margin-bottom: 15px;
    }
    .result {
        background-color: #f5df4d;
        border-radius: 8px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .prediction-value {
        color: #007c91;
        font-weight: bold;
        font-size: 1.2em;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title(" Medical Insurance Predictor")
st.markdown("Predict your medical charges based on personal health information.")

# Input fields
name = st.text_input("Enter your name", key="name", placeholder="Your Name", help="Please enter your full name.", label_visibility="collapsed")
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1, help="Enter your age.")
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, help="Enter your BMI.")
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0, step=1, help="Enter the number of children.")
sex = st.selectbox("Sex", ["male", "female"], key="sex", help="Select your gender")
smoker = st.selectbox("Smoker", ["yes", "no"], key="smoker", help="Are you a smoker?")
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"], key="region", help="Select the region you live in")

# Input DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

# Load and preprocess training data
@st.cache_data
def load_and_train():
    df = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']
    model = LinearRegression()
    model.fit(X, y)
    return model, df_encoded

# Prediction and download Excel
if st.button("Predict Medical Charges"):
    model, df_encoded = load_and_train()
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=df_encoded.drop('charges', axis=1).columns, fill_value=0)
    prediction = model.predict(input_encoded)[0]

    # Show result
    st.markdown(f"""
    <div class="result">
        <h3>Hello, {name}!</h3>
        <p class="prediction-value">💰 Estimated Medical Charges: **${prediction:,.2f}**</p>
    </div>
    """, unsafe_allow_html=True)

    # Prepare Excel Data
    save_df = input_data.copy()
    save_df['Predicted Charges'] = [prediction]
    save_df['Name'] = name
    save_df = save_df[['Name', 'age', 'sex', 'bmi', 'children', 'smoker', 'region', 'Predicted Charges']]

    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        save_df.to_excel(writer, index=False, sheet_name='Prediction')
    output.seek(0)

    # Download button
    st.download_button(
        label="📥 Download Prediction as Excel",
        data=output,
        file_name="prediction.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Optional footer
st.markdown("---")
