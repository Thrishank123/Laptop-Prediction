import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Custom CSS for styling
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f0f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 45px;
        width: 100%;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# Data Loading
@st.cache_data
def load_data():
    df = pd.read_csv('laptop_data_cleaned.csv')
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Select a section", ["Data Exploration", "Price Prediction", "Model Evaluation"])

if option == "Data Exploration":
    st.title("Data Exploration and Insights")
    
    # Dataset Preview
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Price Distribution
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Price'], kde=True, bins=30, color='blue', ax=ax)
    ax.set_title('Laptop Price Distribution')
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation between Features')
    st.pyplot(fig)

    # RAM vs Price
    st.subheader("RAM vs Price")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Ram', y='Price', data=df, color='green', ax=ax)
    ax.set_title('RAM vs Price')
    ax.set_xlabel('RAM (GB)')
    ax.set_ylabel('Price (in Lakhs)')
    st.pyplot(fig)

    # Company vs Price (Boxplot)
    st.subheader("Company vs Price")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Company', y='Price', data=df, ax=ax)
    ax.set_title('Company vs Laptop Price')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    st.subheader("Insights:")
    st.markdown("""
    - **Price Distribution:** Most laptops are in the mid-price range.
    - **RAM vs Price:** There’s a positive correlation between RAM and price.
    - **Company vs Price:** Companies have varied pricing strategies, with premium brands like Apple pricing higher.
    - **Correlation Heatmap:** Identifies key relationships between features, like the link between SSD and Price.
    """)

elif option == "Price Prediction":
    st.title("Laptop Price Prediction")
    st.subheader("Enter Laptop Specifications")

    # User Input for Prediction
    ram = st.slider('RAM (GB)', min_value=4, max_value=64, value=8, step=4)
    weight = st.slider('Weight (KG)', min_value=1.0, max_value=5.0, value=2.5, step=0.1)
    touchscreen = st.selectbox('Touchscreen', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    ips = st.selectbox('IPS Display', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    ppi = st.slider('PPI (Pixels per inch)', min_value=100, max_value=300, value=150, step=10)
    hdd = st.slider('HDD (GB)', min_value=0, max_value=2000, value=500, step=250)
    ssd = st.slider('SSD (GB)', min_value=0, max_value=2000, value=256, step=256)
    company = st.selectbox('Company', sorted(df['Company'].unique()))
    type_name = st.selectbox('Laptop Type', sorted(df['TypeName'].unique()))
    cpu_brand = st.selectbox('CPU Brand', sorted(df['Cpu_brand'].unique()))
    gpu_brand = st.selectbox('GPU Brand', sorted(df['Gpu_brand'].unique()))
    os = st.selectbox('Operating System', sorted(df['Os'].unique()))

    user_data = {
        'Ram': ram,
        'Weight': weight,
        'TouchScreen': touchscreen,
        'Ips': ips,
        'Ppi': ppi,
        'HDD': hdd,
        'SSD': ssd,
        'Company': company,
        'TypeName': type_name,
        'Cpu_brand': cpu_brand,
        'Gpu_brand': gpu_brand,
        'Os': os
    }

    user_df = pd.DataFrame([user_data])

    # Data Preprocessing
    df_encoded = pd.get_dummies(df, columns=['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'Os'], drop_first=True)
    scaler = StandardScaler()
    numerical_features = ['Ram', 'Weight', 'TouchScreen', 'Ips', 'Ppi', 'HDD', 'SSD']
    df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

    user_df = pd.get_dummies(user_df, columns=['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'Os'], drop_first=True)
    user_df[numerical_features] = scaler.transform(user_df[numerical_features])

    # Handle Missing Columns
    missing_cols = set(df_encoded.columns) - set(user_df.columns)
    for col in missing_cols:
        user_df[col] = 0
    user_df = user_df[df_encoded.columns.drop('Price')]

    # Model Selection
    st.subheader("Select Prediction Algorithm")
    algorithm = st.selectbox('Algorithm', ['LightGBM', 'RandomForest', 'LinearRegression'])

    if st.button("Predict Price"):
        if algorithm == 'LightGBM':
            model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05)
        elif algorithm == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()

        # Model Training and Prediction
        X = df_encoded.drop('Price', axis=1)
        y = df_encoded['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        prediction = model.predict(user_df)
        st.write(f"Estimated Laptop Price: ₹{prediction[0] * 10000:,.2f}")

        # Feature Importance for Tree-based models
        if algorithm in ['LightGBM', 'RandomForest']:
            st.subheader("Feature Importance")
            feature_importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)
            st.write(importance_df.head(10))

elif option == "Model Evaluation":
    st.title("Model Evaluation")

    # Data Encoding
    df_encoded = pd.get_dummies(df, columns=['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'Os'], drop_first=True)
    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LightGBM': lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

    st.subheader("Model Comparison")
