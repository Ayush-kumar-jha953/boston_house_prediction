import streamlit as st
import pickle
import numpy as np

# 1. Set up the page title and description
st.set_page_config(page_title="Boston House Price Predictor", layout="centered")
st.title("🏡 Boston House Price Predictor")
st.write("Enter the neighborhood metrics below to predict the median house value (MEDV).")

# 2. Load the saved Linear Regression model
@st.cache_resource  
def load_model():
    with open('linear_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# 3. Create input elements for all 13 features
st.subheader("Neighborhood Features")
col1, col2 = st.columns(2)

with col1:
    crim = st.number_input("CRIM (Per capita crime rate)", value=0.006)
    zn = st.number_input("ZN (Proportion of residential land zoned)", value=18.0)
    indus = st.number_input("INDUS (Proportion of non-retail business acres)", value=2.31)
    chas = st.selectbox("CHAS (Charles River dummy variable)", options=[0, 1], index=0)
    nox = st.number_input("NOX (Nitric oxides concentration)", value=0.538)
    rm = st.number_input("RM (Average number of rooms per dwelling)", value=6.57)
    age = st.number_input("AGE (Proportion of owner-occupied units built prior to 1940)", value=65.2)

with col2:
    dis = st.number_input("DIS (Weighted distances to five Boston employment centres)", value=4.09)
    rad = st.number_input("RAD (Index of accessibility to radial highways)", value=1.0)
    tax = st.number_input("TAX (Full-value property-tax rate per $10,000)", value=296.0)
    ptratio = st.number_input("PTRATIO (Pupil-teacher ratio by town)", value=15.3)
    b = st.number_input("B (1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town)", value=396.9)
    lstat = st.number_input("LSTAT (% lower status of the population)", value=4.98)

# 4. Predict button and output logic
if st.button("Predict House Price", type="primary"):
    input_features = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
    prediction = model.predict(input_features)
    predicted_value = round(float(prediction[0]), 2)
    st.success(f"### 💰 Predicted Median House Value: ${predicted_value:,.2f}k")
