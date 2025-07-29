# 🏠 House Price Predictor (Streamlit App)

This is a machine learning-based web application that predicts property prices in Pakistan based on:

- Property type (House/Flat)
- Purpose (Sale/Rent)
- Location, city, province
- Area, bedrooms, bathrooms
- Geolocation (auto-calculated)
- Inflation-adjusted price

### 🚀 Live Demo

👉 [Click here to use the app](https://house-prediction-app-klepxzcsujemkomkjgppsx.streamlit.app/)

---

### 📦 Features

- Separate models for House/Flat and Rent/Sale
- Location-based encoding and fuzzy matching
- Geolocation from user-entered address
- Loading spinner while predicting
- Inflation-adjusted predictions

---

### 💻 Run Locally

```bash
git clone https://github.com/Hanzala-Naseer/House-Prediction-Streamlit.git
cd House-Prediction-Streamlit
pip install -r requirements.txt
streamlit run app.py
📁 Project Structure
bash
Copy
Edit
├── app.py                  # Main Streamlit app
├── models/                 # Trained .pkl model files
├── utils/frequency_maps.json  # Encoded values for cities, locations, provinces
├── requirements.txt
└── README.md
📌 Note
This app is based on trained models using real estate data from Zameen.com (2020–21), adjusted for inflation.

yaml
Copy
Edit
```
