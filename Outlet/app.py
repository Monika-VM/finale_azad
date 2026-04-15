import streamlit as st
import pandas as pd
import pickle

# Load model & columns
model = pickle.load(open('model.pkl', 'rb'))
cols = pickle.load(open('columns.pkl', 'rb'))

st.title("🛒 Outlet Type Prediction")

# -------- INPUTS --------
item_weight = st.number_input("Item Weight")
item_visibility = st.number_input("Item Visibility")
item_mrp = st.number_input("Item MRP")
outlet_year = st.number_input("Outlet Establishment Year")

# -------- CORRECT ENCODING (MATCH TRAINING) --------

# FIXED mappings (same as training)
fat_map = {'Low Fat': 1, 'Regular': 2}
size_map = {'Small': 1, 'Medium': 2, 'High': 3}
location_map = {'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3}

# ⚠️ IMPORTANT: REPLACE THIS WITH YOUR ACTUAL LabelEncoder classes
item_type_map = {
    'Baking Goods': 0,
    'Breads': 1,
    'Breakfast': 2,
    'Canned': 3,
    'Dairy': 4,
    'Frozen Foods': 5,
    'Fruits and Vegetables': 6,
    'Hard Drinks': 7,
    'Health and Hygiene': 8,
    'Household': 9,
    'Meat': 10,
    'Others': 11,
    'Seafood': 12,
    'Snack Foods': 13,
    'Soft Drinks': 14,
    'Starchy Foods': 15
}

# Dropdowns
item_fat_text = st.selectbox("Item Fat Content", list(fat_map.keys()))
outlet_size_text = st.selectbox("Outlet Size", list(size_map.keys()))
outlet_location_text = st.selectbox("Outlet Location Type", list(location_map.keys()))
item_type_text = st.selectbox("Item Type", list(item_type_map.keys()))

# Convert text → numeric
item_fat = fat_map[item_fat_text]
outlet_size = size_map[outlet_size_text]
outlet_location = location_map[outlet_location_text]
item_type = item_type_map[item_type_text]

# -------- PREDICTION --------
if st.button("Predict"):

    # Create dataframe in SAME order as training
    input_data = pd.DataFrame([[ 
        item_weight,
        item_fat,
        item_visibility,
        item_mrp,
        outlet_year,
        outlet_size,
        outlet_location,
        item_type
    ]], columns=cols)

    prediction = model.predict(input_data)[0]

    # Decode output (check your LabelEncoder if needed)
    output_map = {
        0: "Grocery Store",
        1: "Supermarket Type1",
        2: "Supermarket Type2",
        3: "Supermarket Type3"
    }

    st.success(f"✅ Prediction: {output_map.get(prediction, prediction)}")