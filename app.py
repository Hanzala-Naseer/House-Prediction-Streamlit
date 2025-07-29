# import streamlit as st
# import pickle
# import numpy as np
# import json
# import os
# from geopy.geocoders import Nominatim
# from collections import defaultdict

# # === Load frequency maps ===
# with open("utils/frequency_maps.json", "r", encoding="utf-8") as f:
#     freq_maps = json.load(f)

# # === Province to Cities ===
# province_to_cities = {
#     "Punjab": ["Lahore", "Rawalpindi", "Faisalabad"],
#     "Sindh": ["Karachi"],
#     "Islamabad Capital": ["Islamabad"]
# }

# # === Build City to Locations ===
# city_to_locations = defaultdict(list)
# for location in freq_maps["location"].keys():
#     for city in freq_maps["city"].keys():
#         if city.lower() in location.lower():
#             city_to_locations[city].append(location)

# # === Model mapping ===
# model_map = {
#     ("Flat", "For Sale"): "flat_sale_model.pkl",
#     ("Flat", "For Rent"): "flat_rent_model.pkl",
#     ("House", "For Sale"): "house_sale_model.pkl",
#     ("House", "For Rent"): "house_rent_model.pkl",
# }

# # === Geocoding ===
# geolocator = Nominatim(user_agent="price-predictor")
# def get_coordinates(address):
#     try:
#         location = geolocator.geocode(address,timeout=10)
#         if location:
#             return location.latitude, location.longitude
#     except Exception as e:
#         st.error(f"Geocoding error: {e}")
#     return None, None

# # === Streamlit UI ===
# st.set_page_config(page_title="Property Price Predictor", layout="centered")
# st.markdown("""
#     <style>
#     body {
#         background-color: #f8f9fa;
#         font-family: 'Segoe UI', sans-serif;
#     }
#     .stButton button {
#         background-color: #003366;
#         color: white;
#         font-weight: bold;
#         border-radius: 5px;
#         padding: 10px 16px;
#     }
#     .stButton button:hover {
#         background-color: #004080;
#     }
#     .stSelectbox > div, .stNumberInput input, .stTextInput input {
#         border-radius: 6px !important;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.title("🏠 Property Price Predictor")

# # === User Inputs ===
# property_type = st.selectbox("Property Type", ["Flat", "House"])
# purpose = st.selectbox("Purpose", ["For Sale", "For Rent"])

# province = st.selectbox("Province", sorted(province_to_cities.keys()))
# valid_cities = province_to_cities.get(province, [])
# city = st.selectbox("City", valid_cities if valid_cities else ["Select province first"])

# valid_locations = city_to_locations.get(city, [])
# location = st.selectbox("Location", sorted(valid_locations) if valid_locations else ["Select city first"])

# address = st.text_input("Complete Address", placeholder="e.g. House 12, DHA Phase 5, Lahore")

# bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, step=1)
# baths = st.number_input("Bathrooms", min_value=1, max_value=20, step=1)
# area_marla = st.number_input("Area (in Marla)", min_value=1.0, max_value=200.0)

# # # === Predict Button ===
# # if st.button("Predict Price"):
# #     latitude, longitude = get_coordinates(address)

# #     if latitude is None or longitude is None:
# #         st.error("❌ Could not extract coordinates from the address.")
# #     else:
# #         area_sqft = area_marla * 272.25
# #         model_key = (property_type, purpose)
# #         model_file = model_map.get(model_key)

# #         if model_file and os.path.exists(f"models/{model_file}"):
# #             with open(f"models/{model_file}", "rb") as f:
# #                 model, scaler = pickle.load(f)

# #             loc_enc = freq_maps["location"].get(location, 0)
# #             city_enc = freq_maps["city"].get(city, 0)
# #             prov_enc = freq_maps["province_name"].get(province, 0)

# #             area_per_bed = area_sqft / (bedrooms + 1)
# #             area_baths_product = area_sqft * baths

# #             features = np.array([[bedrooms, baths, area_sqft, latitude, longitude,
# #                                   loc_enc, city_enc, prov_enc,
# #                                   area_per_bed, area_baths_product]])

# #             features_scaled = scaler.transform(features)

# #             log_price_pred = model.predict(features_scaled)[0]
# #             price = np.expm1(log_price_pred)

# #             st.success(f"💰 Estimated Price: PKR {price:,.0f}")
# #         else:
# #             st.error("❌ Model not found or not trained.")
# # === Predict Button ===
# if st.button("Predict Price"):
#     latitude, longitude = get_coordinates(address)

#     if latitude is None or longitude is None:
#         st.error("❌ Could not extract coordinates from the address.")
#     else:
#         area_sqft = area_marla * 272.25
#         model_key = (property_type, purpose)
#         model_file = model_map.get(model_key)

#         if model_file and os.path.exists(f"models/{model_file}"):
#             with open(f"models/{model_file}", "rb") as f:
#                 model, scaler = pickle.load(f)

#             loc_enc = freq_maps["location"].get(location, 0)
#             city_enc = freq_maps["city"].get(city, 0)
#             prov_enc = freq_maps["province_name"].get(province, 0)

#             area_per_bed = area_sqft / (bedrooms + 1)
#             area_baths_product = area_sqft * baths

#             features = np.array([[bedrooms, baths, area_sqft, latitude, longitude,
#                                   loc_enc, city_enc, prov_enc,
#                                   area_per_bed, area_baths_product]])

#             features_scaled = scaler.transform(features)

#             log_price_pred = model.predict(features_scaled)[0]
#             price = np.expm1(log_price_pred)

#             # === Inflation adjustment ===
#             inflation_factor = 2.0
#             adjusted_price = price * inflation_factor

#             st.success(f"💰 Estimated Price (Adjusted for 2024–25): PKR {adjusted_price:,.0f}")
#             st.caption(f"Note: Base prediction was PKR {price:,.0f} (based on 2020–21 data, adjusted x{inflation_factor})")
#         else:
#             st.error("❌ Model not found or not trained.")


import streamlit as st
import pickle
import numpy as np
import json
import os
from geopy.geocoders import Nominatim
from collections import defaultdict

# === Load frequency maps ===
with open("utils/frequency_maps.json", "r", encoding="utf-8") as f:
    freq_maps = json.load(f)

# === Province to Cities ===
province_to_cities = {
    "Punjab": ["Lahore", "Rawalpindi", "Faisalabad"],
    "Sindh": ["Karachi"],
    "Islamabad Capital": ["Islamabad"]
}

# === Build City to Locations ===
city_to_locations = defaultdict(list)
for location in freq_maps["location"].keys():
    for city in freq_maps["city"].keys():
        if city.lower() in location.lower():
            city_to_locations[city].append(location)

# === Model mapping ===
model_map = {
    ("Flat", "For Sale"): "flat_sale_model.pkl",
    ("Flat", "For Rent"): "flat_rent_model.pkl",
    ("House", "For Sale"): "house_sale_model.pkl",
    ("House", "For Rent"): "house_rent_model.pkl",
}

# === Geocoding ===
geolocator = Nominatim(user_agent="price-predictor")
def get_coordinates(address):
    try:
        location = geolocator.geocode(address,timeout=15)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        st.error(f"Geocoding error: {e}")
    return None, None

# === Streamlit UI ===
st.set_page_config(page_title="Property Price Predictor", layout="centered")
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton button {
        background-color: #003366;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 16px;
    }
    .stButton button:hover {
        background-color: #004080;
    }
    .stSelectbox > div, .stNumberInput input, .stTextInput input {
        border-radius: 6px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏠 Property Price Predictor")

# === User Inputs ===
property_type = st.selectbox("Property Type", ["Flat", "House"])
purpose = st.selectbox("Purpose", ["For Sale", "For Rent"])

province = st.selectbox("Province", sorted(province_to_cities.keys()))
valid_cities = province_to_cities.get(province, [])
city = st.selectbox("City", valid_cities if valid_cities else ["Select province first"])

valid_locations = city_to_locations.get(city, [])
location = st.selectbox("Location", sorted(valid_locations) if valid_locations else ["Select city first"])

address = st.text_input("Complete Address", placeholder="e.g. House 12, DHA Phase 5, Lahore")

bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, step=1)
baths = st.number_input("Bathrooms", min_value=1, max_value=20, step=1)
area_marla = st.number_input("Area (in Marla)", min_value=1.0, max_value=200.0)
from difflib import get_close_matches

# === Helper: Fuzzy match location from user input ===
def get_best_location_match(user_input_loc, known_locations):
    matches = get_close_matches(user_input_loc.lower(), [loc.lower() for loc in known_locations], n=1, cutoff=0.6)
    if matches:
        for loc in known_locations:
            if loc.lower() == matches[0]:
                return loc
    return None

# === Predict Button ===
# if st.button("Predict Price"):
#     latitude, longitude = get_coordinates(address)

#     if latitude is None or longitude is None:
#         st.error("❌ Could not extract coordinates from the address.")
#     else:
#         area_sqft = area_marla * 272.25
#         model_key = (property_type, purpose)
#         model_file = model_map.get(model_key)

#         if model_file and os.path.exists(f"models/{model_file}"):
#             with open(f"models/{model_file}", "rb") as f:
#                 model, scaler = pickle.load(f)

#             # ✅ Improved location encoding with fuzzy match
#             matched_location = get_best_location_match(location, freq_maps["location"].keys())
#             loc_enc = freq_maps["location"].get(matched_location, 0)

#             city_enc = freq_maps["city"].get(city, 0)
#             prov_enc = freq_maps["province_name"].get(province, 0)

#             area_per_bed = area_sqft / (bedrooms + 1)
#             area_baths_product = area_sqft * baths

#             features = np.array([[bedrooms, baths, area_sqft, latitude, longitude,
#                                   loc_enc, city_enc, prov_enc,
#                                   area_per_bed, area_baths_product]])

#             features_scaled = scaler.transform(features)

#             log_price_pred = model.predict(features_scaled)[0]
#             price = np.expm1(log_price_pred)

#             # === Inflation adjustment ===
#             inflation_factor = 1.5
#             adjusted_price = price * inflation_factor

#             st.success(f"💰 Estimated Price (Adjusted for 2024–25): PKR {adjusted_price:,.0f}")
#             st.caption(f"Note: Base prediction was PKR {price:,.0f} (based on 2020–21 data, adjusted x{inflation_factor})")
#         else:
#             st.error("❌ Model not found or not trained.")

if st.button("Predict Price"):
    with st.spinner("Predicting... Please wait."):
        latitude, longitude = get_coordinates(address)

        if latitude is None or longitude is None:
            st.error("❌ Could not extract coordinates from the address.")
        else:
            area_sqft = area_marla * 272.25
            model_key = (property_type, purpose)
            model_file = model_map.get(model_key)

            if model_file and os.path.exists(f"models/{model_file}"):
                with open(f"models/{model_file}", "rb") as f:
                    model, scaler = pickle.load(f)

                # ✅ Improved location encoding with fuzzy match
                matched_location = get_best_location_match(location, freq_maps["location"].keys())
                loc_enc = freq_maps["location"].get(matched_location, 0)

                city_enc = freq_maps["city"].get(city, 0)
                prov_enc = freq_maps["province_name"].get(province, 0)

                area_per_bed = area_sqft / (bedrooms + 1)
                area_baths_product = area_sqft * baths

                features = np.array([[bedrooms, baths, area_sqft, latitude, longitude,
                                      loc_enc, city_enc, prov_enc,
                                      area_per_bed, area_baths_product]])

                features_scaled = scaler.transform(features)
                log_price_pred = model.predict(features_scaled)[0]
                price = np.expm1(log_price_pred)

                # === Inflation adjustment ===
                inflation_factor = 1.5
                adjusted_price = price * inflation_factor

                st.success(f"💰 Estimated Price (Adjusted for 2024–25): PKR {adjusted_price:,.0f}")
                st.caption(f"Note: Base prediction was PKR {price:,.0f} (based on 2020–21 data, adjusted x{inflation_factor})")
            else:
                st.error("❌ Model not found or not trained.")
