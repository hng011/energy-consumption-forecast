import streamlit as st
import pandas as pd
import requests
import joblib

models = [
    "https://github.com/hng011/wok/raw/refs/heads/dev/models/model_elastic_fnb1.joblib",
    "https://github.com/hng011/wok/raw/refs/heads/dev/models/model_linreg_fnb1.joblib"
]

scaler = "https://github.com/hng011/wok/raw/refs/heads/dev/models/scaler_standardscaler_fnb1.joblib"


def fetch_model(endpoint, file_name):
    try:
        response = requests.get(endpoint, stream=True)
        response.raise_for_status()
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")
        st.stop()

    # Load the model
    try:
        return joblib.load(file_name)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        
        
st.image("https://raw.githubusercontent.com/hng011/wok/refs/heads/main/assets/banner_linreg.png")
st.title("Energy Consumption Prediction")
st.markdown("### Enter the details below to predict the energy consumption.")


list_bt = ["Residential", "Commercial", "Industrial"]
list_dw = ["Weekday", "Weekend"]


building_type = st.selectbox("Building Type", list_bt)
square_footage = st.number_input("Square Footage", min_value=0, max_value=100000, value=0)
occupants = st.number_input("Number of Occupants", min_value=0, max_value=1000, value=0)
appliances = st.number_input("Appliances Used", min_value=0, max_value=1000, value=0)
temperature = st.number_input("Average Temperature (°C)", min_value=00.0, max_value=100.0, value=0.0)
day_of_week = st.selectbox("Day of Week", list_dw)


# Choosing Model
list_model = ["LinearRegression", "ElasticNet"]
choosed_model = st.selectbox("Model", list_model)


# # Choosing Scaler
# list_scaler = ["StandardScaler", "MinMaxScaler"]
# choosed_scaler = st.selectbox("Scaler", list_scaler)


data_cols = [
        "Square Footage", 
        "Number of Occupants", 
        "Appliances Used", 
        "Average Temperature", 
        "Building Type_Commercial",
        "Building Type_Industrial",
        "Building Type_Residential",
        "Day of Week_Weekday",
        "Day of Week_Weekend"
]


input_data = pd.DataFrame(
    data=[ [square_footage, occupants, appliances, temperature, 0.0, 0.0, 0.0, 0.0, 0.0] ], 
    columns=data_cols
)


input_data = input_data.astype("float64")


# Encode
if building_type == list_bt[0]: input_data[data_cols[6]] = 1.0
elif building_type == list_bt[1]: input_data[data_cols[4]] = 1.0
else: input_data[data_cols[5]] = 1.0

if day_of_week == list_dw[0]: input_data[data_cols[-2]] = 1.0
else: input_data[data_cols[-1]] = 1.0


# scaler
scaler = fetch_model(endpoint=scaler, file_name="scaler.joblib")
input_data = scaler.transform(input_data)


# Predict Energy Consumption
if st.button("Predict Energy Consumption"):
    model = fetch_model(endpoint=models[0] if choosed_model == list_model[1] else models[1], file_name="model.joblib")
    predicted_energy = model.predict(input_data)[0]
    st.subheader("🔮 Prediction Result")
    st.write(f"### Predicted Energy Consumption (kWh): {predicted_energy:.2f}")