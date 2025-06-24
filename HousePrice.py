import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
df = pd.read_csv("ames iowa housing.csv")
X = df[['GrLivArea', 'LotArea', 'BedroomAbvGr', 'GarageCars']]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
xgb_model.fit(X_train, y_train)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Streamlit UI
st.title("🏡 House Price Predictor – House Dataset")

st.sidebar.header("🏗️ Property Details")
gr_liv_area = st.sidebar.slider("Living Area (sq ft)", 500, 5000, 1500)
lot_area = st.sidebar.slider("Lot Size (sq ft)", 2000, 20000, 8000)
bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, 3)
garage = st.sidebar.slider("Garage Capacity (Cars)", 0, 4, 2)

# Create input DataFrame
input_data = pd.DataFrame([[gr_liv_area, lot_area, bedrooms, garage]],
                          columns=X.columns)

# Predict
xgb_pred = xgb_model.predict(input_data)[0]
lr_pred = lr_model.predict(input_data)[0]

# Show Results
st.subheader("💡 Predicted Prices")
st.success(f"📈 XGBoost Model: ${xgb_pred:,.0f}")
st.info(f"📉 Linear Regression: ${lr_pred:,.0f}")

# RMSE Comparison
from sklearn.metrics import mean_squared_error

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_model.predict(X_test)))
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_model.predict(X_test)))

st.write("### 📊 Model Accuracy")
st.metric(label="XGBoost RMSE", value=f"${xgb_rmse:,.0f}")
st.metric(label="Linear Regression RMSE", value=f"${lr_rmse:,.0f}")

