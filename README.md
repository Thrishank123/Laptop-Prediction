# Laptop Price Prediction Streamlit App

This project is a web application built using **Streamlit** to predict laptop prices based on user input specifications and visualize insights from a dataset of laptops. The app allows users to explore data, predict laptop prices, and evaluate different machine learning models.

## Overview

This app enables users to interact with a dataset of laptops and predict prices using machine learning algorithms such as **LightGBM**, **RandomForest**, and **Linear Regression**. Additionally, it provides data visualizations like distribution plots, scatter plots, and box plots to help users gain insights from the dataset.

## App Features

1. **Data Exploration**:
   - Preview the dataset.
   - Visualize price distribution, RAM vs. Price, and Company vs. Price comparisons.

2. **Laptop Price Prediction**:
   - Input various laptop specifications (RAM, Weight, Touchscreen, etc.).
   - Predict laptop prices using one of three models: LightGBM, RandomForest, or Linear Regression.

3. **Model Evaluation**:
   - Compare performance metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score for different models.
   - Visualize actual vs predicted prices using LightGBM.

## Dataset

The dataset used in this project is a cleaned version of a laptop dataset containing the following columns:
- **Price**: Price of the laptop.
- **Company**: Laptop brand (e.g., Dell, Apple, HP).
- **Ram**: Amount of RAM in GB.
- **Weight**: Weight of the laptop in kg.
- **TouchScreen**: Boolean indicating whether the laptop has a touchscreen.
- **Ips**: Boolean indicating whether the laptop has an IPS display.
- **Ppi**: Pixels per inch of the screen.
- **HDD**: Storage capacity in HDD.
- **SSD**: Storage capacity in SSD.
- **Cpu_brand**: CPU manufacturer.
- **Gpu_brand**: GPU manufacturer.
- **Os**: Operating system installed.

## Machine Learning Models

Three models are used for prediction:
- **LightGBM**: A high-performance, gradient boosting framework.
- **RandomForest**: A versatile ensemble learning method using decision trees.
- **Linear Regression**: A simple and interpretable regression model.

## Usage

1. **Data Exploration**:
   - View insights from the dataset, such as price distribution, RAM vs. price correlation, and price comparison across companies.

2. **Price Prediction**:
   - Select laptop specifications using sliders and dropdowns.
   - Predict the price based on the chosen machine learning model.

3. **Model Evaluation**:
   - Compare different models and their performance metrics.
   - Visualize the actual vs predicted prices.

## Installation

### Prerequisites

- Python 3.x
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, lightgbm, streamlit
