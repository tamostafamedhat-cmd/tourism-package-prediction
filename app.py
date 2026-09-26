import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="centered")
st.title("Wellness Tourism Package Predictor")
st.write("Predict whether a customer is likely to purchase the Wellness Tourism Package.")

model = joblib.load("outputs/best_model.joblib")

with st.form("customer_form"):
    age = st.number_input("Age", 18, 100, 35)
    contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city = st.selectbox("City Tier", [1, 2, 3])
    duration = st.number_input("Duration of Pitch", 1, 120, 15)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    persons = st.number_input("Number of Persons Visiting", 1, 10, 3)
    followups = st.number_input("Number of Followups", 0, 10, 3)
    product = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    stars = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital = st.selectbox("Marital Status", ["Single", "Unmarried", "Married", "Divorced"])
    trips = st.number_input("Number of Trips", 0, 30, 2)
    passport = st.selectbox("Passport", [0, 1])
    pitch_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    own_car = st.selectbox("Own Car", [0, 1])
    children = st.number_input("Number of Children Visiting", 0, 5, 1)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly Income", 10000, 100000, 25000)
    submitted = st.form_submit_button("Predict")

if submitted:
    data = pd.DataFrame([{
        "Age": age, "TypeofContact": contact, "CityTier": city, "DurationOfPitch": duration,
        "Occupation": occupation, "Gender": gender, "NumberOfPersonVisiting": persons,
        "NumberOfFollowups": followups, "ProductPitched": product, "PreferredPropertyStar": stars,
        "MaritalStatus": marital, "NumberOfTrips": trips, "Passport": passport,
        "PitchSatisfactionScore": pitch_score, "OwnCar": own_car,
        "NumberOfChildrenVisiting": children, "Designation": designation, "MonthlyIncome": income
    }])
    probability = model.predict_proba(data)[0, 1]
    st.metric("Purchase Probability", f"{probability:.1%}")
    st.success("Likely Buyer - prioritize for campaign") if probability >= 0.5 else st.info("Not Likely Buyer - keep in nurture segment")
