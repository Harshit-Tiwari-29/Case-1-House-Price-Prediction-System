import streamlit as st
import requests

st.set_page_config(page_title="House Price AI", page_icon="🏡")

st.title("🏡 AI Real Estate Estimator")
st.markdown("Enter property details to predict the market value.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        size = st.number_input("Size (Sq Ft)", 500, 10000, 2000)
        bedrooms = st.slider("Bedrooms", 1, 10, 3)
        bathrooms = st.slider("Bathrooms", 1, 10, 2)
        year_built = st.number_input("Year Built", 1800, 2025, 2000)

    with col2:
        location = st.selectbox("Location", ["CityA", "CityB", "CityC", "CityD"])
        prop_type = st.selectbox("Property Type", ["Single Family", "Condominium", "Townhouse"])
        condition = st.selectbox("Condition", ["New", "Good", "Fair", "Poor"])

    submitted = st.form_submit_button("Predict Price 🚀")

if submitted:
    api_url = "https://case-1-house-price-prediction-system.onrender.com/predict"
    
    payload = {
        "size_sqft": size,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "location": location,
        "property_type": prop_type,
        "condition": condition
    }
    
    try:
        with st.spinner("Calculating value..."):
            response = requests.post(api_url, json=payload)
            
        if response.status_code == 200:
            price = response.json()["predicted_price"]
            st.success(f"💰 Estimated Price: **${price:,.2f}**")
        else:
            st.error(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Backend. Run 'uvicorn backend.main:app' first.")
