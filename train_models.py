import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import json


# === Load & Clean Data ===
df = pd.read_csv("data/Entities.csv")

# Drop irrelevant or unhelpful columns
drop_cols = [
    "Unnamed: 0", "page_url", "property_id", "location_id", "agency", "agent", 
    "date_added", "title", "description", "floor", "floors", "street", "parking_spaces"
]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Drop rows with missing critical values
df.dropna(subset=["price", "Total_Area", "bedrooms", "baths", "latitude", "longitude"], inplace=True)

# Keep only target property types and purposes
df = df[df["property_type"].isin(["Flat", "House"]) & df["purpose"].isin(["For Sale", "For Rent"])]

# === Outlier Removal ===
df = df[df["price"].between(df["price"].quantile(0.02), df["price"].quantile(0.98))]
df = df[df["Total_Area"].between(df["Total_Area"].quantile(0.02), df["Total_Area"].quantile(0.98))]

# === Frequency Encoding for Categoricals ===
for col in ["location", "city", "province_name"]:
    freqs = df[col].value_counts().to_dict()
    df[col + "_encoded"] = df[col].map(freqs)

# === Feature Engineering ===
df["area_per_bed"] = df["Total_Area"] / (df["bedrooms"] + 1)
df["area_baths_product"] = df["Total_Area"] * df["baths"]

# === Final Selected Features ===
features = [
    "bedrooms", "baths", "Total_Area", "latitude", "longitude",
    "location_encoded", "city_encoded", "province_name_encoded",
    "area_per_bed", "area_baths_product"
]

# === Save Models Here ===
os.makedirs("models", exist_ok=True)

# === Define Target Models ===
target_models = {
    ("House", "For Sale"): "house_sale_model",
    ("House", "For Rent"): "house_rent_model",
    ("Flat", "For Sale"): "flat_sale_model",
    ("Flat", "For Rent"): "flat_rent_model",
}

# === Training Loop ===
for (ptype, purpose), model_name in target_models.items():
    group = df[(df["property_type"] == ptype) & (df["purpose"] == purpose)]
    if len(group) < 500:
        print(f"[SKIP] Not enough data for {model_name} ({len(group)} rows)")
        continue

    X = group[features]
    y = np.log1p(group["price"])  # log transform to stabilize

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # XGBoost model
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"[{ptype} - {purpose}] R²: {r2:.4f} ({r2*100:.2f}%) | RMSE: {rmse:.2f} | Samples: {len(group)}")

    if r2 >= 0.80:
        with open(f"models/{model_name}.pkl", "wb") as f:
            pickle.dump((model, scaler), f)
        print(f"✅ Model saved: {model_name}.pkl\n")
    else:
        print(f"❌ Skipped saving {model_name} (R² < 0.80)\n")



# Generate frequency maps from the training DataFrame
def create_frequency_maps(df, columns):
    freq_maps = {}
    for col in columns:
        freq_map = df[col].value_counts().to_dict()
        freq_maps[col] = freq_map
    return freq_maps

# Columns to frequency encode
freq_columns = ["location", "city", "province_name"]

# Generate the frequency mappings
frequency_maps = create_frequency_maps(df, freq_columns)

# Save to JSON file
import json
with open("utils/frequency_maps.json", "w", encoding="utf-8") as f:
    json.dump(frequency_maps, f, ensure_ascii=False, indent=2)

print("✅ Frequency maps saved to utils/frequency_maps.json")
