import streamlit as st
import pandas as pd
import requests
import pickle

ep_model = "https://github.com/hng011/wok/raw/refs/heads/main/models/elasticnet_model.pkl"
ep_scaler = "https://github.com/hng011/wok/raw/refs/heads/main/models/scaler.pkl"

def fetch_model(ep, file_name):
    try:
        response = requests.get(ep, stream=True)
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
        with open(file_name, "rb") as f:
            return  pickle.load(f)            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        
        
st.title("Energy Consumption Prediction")
st.markdown("### Enter the details below to predict the energy consumption.")


building_type = st.selectbox("Building Type", ["Residential", "Commercial", "Industrial"])
square_footage = st.number_input("Square Footage", min_value=0, max_value=100000)
occupants = st.number_input("Number of Occupants", min_value=1, max_value=1000)
appliances = st.number_input("Appliances Used", min_value=1, max_value=1000)
temperature = st.number_input("Average Temperature (°C)", min_value=00.0, max_value=100.0)
day_of_week = st.selectbox("Day of Week", ["Weekday", "Weekend"])


building_type_map = {'Residential': 0.347, 'Commercial': 0.336, 'Industrial': 0.317}
day_of_week_map = {'Weekday': 0.507, 'Weekend': 0.493}


input_data = pd.DataFrame(
    data=[[building_type_map[building_type], square_footage, occupants, appliances, temperature, day_of_week_map[day_of_week]]], 
    columns=["Building Type", "Square Footage", "Number of Occupants", "Appliances Used", "Average Temperature", "Day of Week"]
)

input_data = input_data.astype("float64")

# scaler
scaler = fetch_model(ep=ep_scaler, file_name="scaler.pkl")
input_data = scaler.transform(input_data)


# Predict Energy Consumption
if st.button("Predict Energy Consumption"):
    model = fetch_model(ep=ep_model, file_name="model.pkl")
    predicted_energy = model.predict(input_data)[0]
    st.subheader("🔮 Prediction Result")
    st.write(f"### Predicted Energy Consumption: {predicted_energy:.2f}")