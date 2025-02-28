import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Title and Description
st.title("Real Estate Price Prediction App")
st.markdown("""
This app predicts real estate prices using a Linear Regression model.  
Upload a CSV file containing 'Area', 'Bedrooms', 'Location', and 'Price' columns.
""")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    # Read the CSV
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    
    # Check for necessary columns
    if {'Area', 'Bedrooms', 'Location', 'Price'}.issubset(df.columns):
        # Encode Categorical Data
        le = LabelEncoder()
        df['Location'] = le.fit_transform(df['Location'])
        
        # Step 2: Data Visualization
        st.subheader("Price Distribution")
        plt.figure(figsize=(10, 5))
        plt.hist(df['Price'], bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.title('Price Distribution')
        st.pyplot(plt)
        
        st.subheader("Price vs Area")
        plt.figure(figsize=(10, 5))
        plt.scatter(df['Area'], df['Price'], color='green')
        plt.xlabel('Area (sq ft)')
        plt.ylabel('Price')
        plt.title('Price vs Area')
        st.pyplot(plt)
        
        # Step 3: Train-Test Split
        X = df[['Area', 'Bedrooms', 'Location']]
        y = df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Step 4: Model Training
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Model Evaluation
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        st.subheader("Model Performance")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        
        # Step 5: Prediction
        st.subheader("Make a Prediction")
        area = st.number_input("Enter Area (in sq ft)", min_value=100, max_value=10000, value=1000)
        bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, max_value=10, value=2)
        location = st.selectbox("Select Location", le.classes_)
        location_encoded = le.transform([location])[0]
        
        prediction = model.predict(np.array([[area, bedrooms, location_encoded]]))
        st.write(f"Predicted Price: ${prediction[0]:,.2f}")
    else:
        st.error("CSV must have 'Area', 'Bedrooms', 'Location', and 'Price' columns.")
else:
    st.info("Please upload a CSV file to get started.")
