import streamlit as st
import numpy as np
import pickle
import requests

API_KEY = "da00247b37a4b1ea98bee8020722ebeb"

# Load files
model = pickle.load(
    open("crop_model.pkl","rb")
)

scaler = pickle.load(
    open("scaler.pkl","rb")
)

encoder = pickle.load(
    open("encoder.pkl","rb")
)

# Weather function
def get_weather(city):

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}"
        f"&appid={API_KEY}"
        f"&units=metric"
    )

    response = requests.get(url)

    return response.json()

# Title
st.title("Agritech Connect")

st.subheader(
    "Weather-Enabled Crop Recommendation"
)

# City input
city = st.text_input(
    "Enter City"
)

# Weather
if city:

    weather = get_weather(city)

    temperature = weather["main"]["temp"]
    humidity = weather["main"]["humidity"]

    st.write(
        f"Temperature: {temperature} °C"
    )

    st.write(
        f"Humidity: {humidity}%"
    )

# Soil inputs
N = st.number_input(
    "Nitrogen"
)

P = st.number_input(
    "Phosphorus"
)

K = st.number_input(
    "Potassium"
)

ph = st.number_input(
    "pH"
)

rainfall = st.number_input(
    "Rainfall"
)

# Prediction
if st.button("Predict Crop"):

    input_data = np.array([
        [
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]
    ])

    input_scaled = scaler.transform(
        input_data
    )

    prediction = model.predict(
        input_scaled
    )

    crop_name = encoder.inverse_transform(
        prediction
    )

    st.success(
        f"Recommended Crop: {crop_name[0]}"
    )
    