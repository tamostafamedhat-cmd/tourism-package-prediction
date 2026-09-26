if submitted:
    data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": contact,
        "CityTier": city,
        "DurationOfPitch": duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "NumberOfFollowups": followups,
        "ProductPitched": product,
        "PreferredPropertyStar": stars,
        "MaritalStatus": marital,
        "NumberOfTrips": trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income
    }])

    probability = model.predict_proba(data)[0, 1]

    st.metric("Purchase Probability", f"{probability:.1%}")

    if probability >= 0.5:
        st.success("Likely Buyer - prioritize for campaign")
    else:
        st.info("Not Likely Buyer - keep in nurture segment")
