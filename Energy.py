# Final Working code 1 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

# Set up the page
st.set_page_config(page_title="Energy Consumption Analysis", layout="wide")

# Sidebar navigation
st.sidebar.title("Energy Consumption Analysis")
page = st.sidebar.radio(
    "Select Page:",
    ["📊 Overview", "📦 Outliers", "📊 Skewness Analysis", "🤖 Models", "🔮 Predict"]
)

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\extracted\household_power_consumption.txt", sep=';', na_values='?', low_memory=False)

    # Parse date and time
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    df.dropna(inplace=True)

    # Convert data types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add time-based features
    df['Year'] = df.index.year.astype('int32')
    df['Month'] = df.index.month.astype('int32')
    df['Day'] = df.index.day.astype('int32')
    df['Hour'] = df.index.hour.astype('int32')
    df['Minute'] = df.index.minute.astype('int32')
    df['Weekday'] = df.index.day_name()

    # Encode the 'Weekday' column with Label Encoding
    le = LabelEncoder()
    df['Weekday'] = le.fit_transform(df['Weekday'])

    return df, le  # Returning the LabelEncoder for reuse in prediction

# Load data and LabelEncoder
df, le = load_data()

# 📊 Overview
if page == "📊 Overview":
    st.title("Dataset Overview")
    st.write("First 5 Rows")
    st.write(df.head())

    st.write("Summary Statistics")
    st.write(df.describe())

    st.write("Data Types and Memory Usage")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.write("Correlation Heatmap (Selected Columns)")
    selected_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 
                     'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df[selected_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
# 📦 Outliers
elif page == "📦 Outliers":
    st.title("Outlier Detection and Capping")
    cols_to_check = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                     'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    # Function to detect outliers using IQR method
    def detect_outliers(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return df[(df[col] < lower) | (df[col] > upper)]

    # Function to cap outliers
    def cap_outliers(df, col):
        lower = df[col].quantile(0.05)
        upper = df[col].quantile(0.95)
        df[col] = np.where(df[col] < lower, lower, np.where(df[col] > upper, upper, df[col]))

    # Display Outliers and Boxplots before capping
    st.write("### Boxplots Before Capping")
    num_cols = 3  # Define how many columns per row
    num_rows = (len(cols_to_check) + num_cols - 1) // num_cols  # Calculate number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 6 * num_rows))
    axes = axes.flatten()

    # Plot boxplots for each column
    for i, col in enumerate(cols_to_check):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Boxplot for {col}')
        axes[i].tick_params(axis='x', labelrotation=45)
        axes[i].set_xlabel("")
    
    # Hide extra axes if there are any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

    st.write("### Number of Outliers")
    for col in cols_to_check:
        outliers = detect_outliers(df, col).shape[0]
        st.write(f"{col}: {outliers} outliers")

    for col in cols_to_check:
        cap_outliers(df, col)

    # Display Boxplots After Capping
    st.write("### Boxplots After Capping")
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 6 * num_rows))
    axes = axes.flatten()

    # Plot boxplots for each column after capping
    for i, col in enumerate(cols_to_check):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Capped {col}')
        axes[i].tick_params(axis='x', labelrotation=45)
        axes[i].set_xlabel("")

    # Hide extra axes if there are any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

# 📊 Skewness Analysis
elif page == "📊 Skewness Analysis":
    st.title("Skewness Analysis")
    cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    skew_vals = df[cols].skew()
    
    # Calculate number of rows and columns for subplots dynamically based on the number of columns
    num_cols = 3  # Define how many columns per row
    num_rows = (len(cols) + num_cols - 1) // num_cols  # Calculate number of rows needed

    # Create the subplots grid
    plt.figure(figsize=(16, 6 * num_rows))
    for i, col in enumerate(cols, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.histplot(df[col], kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.text(0.5, 0.95, f'Skewness: {skew_vals[col]:.2f}',
                 horizontalalignment='center', verticalalignment='top',
                 transform=plt.gca().transAxes, fontsize=12, color='red', weight='bold')
    
    plt.tight_layout()
    st.pyplot(plt)

# 🤖 Models
elif page == "🤖 Models":
    st.title("Model Comparison (Manually Added Metrics)")
    model_data = {
        "Model": ["Random Forest", "Decision Tree", "Ridge Regression",
                  "Linear Regression", "Lasso Regression", "K-Nearest Neighbors"],
        "RMSE": [0.0275, 0.0384, 0.0416, 0.0416, 0.0487, 0.2066],
        "MAE": [0.0114, 0.0145, 0.0266, 0.0266, 0.0325, 0.1272],
        "R²": [0.9992, 0.9985, 0.9983, 0.9983, 0.9976, 0.9573]
    }
    results_df = pd.DataFrame(model_data).sort_values(by="RMSE")
    st.write("Model Performance Metrics")
    st.dataframe(results_df)

    st.write("### Best Models Based on Each Metric:")
    st.write(f"**Best RMSE**: {results_df.loc[results_df['RMSE'].idxmin(), 'Model']}")
    st.write(f"**Best MAE**: {results_df.loc[results_df['MAE'].idxmin(), 'Model']}")
    st.write(f"**Best R²**: {results_df.loc[results_df['R²'].idxmax(), 'Model']}")

# 🔮 Predict
elif page == "🔮 Predict":
    st.title("Predict Global Active Power")

    # Prepare the feature matrix X and target variable y
    X = df.drop(columns=['Global_active_power'])
    y = df['Global_active_power']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': [10],       # Start with a lower number to speed up evaluation
        'max_depth': [10],           # Limit depth to avoid overfitting
        'min_samples_split': [5],    # Limit splits for fewer, but more meaningful splits
        'min_samples_leaf': [2],     # More restrictive leaf nodes
        'max_features': ['sqrt']    # Test square-root feature selection
    }

    # Initialize RandomForestRegressor and RandomizedSearchCV
    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(rf, param_dist, n_iter=5, cv=3, random_state=42, n_jobs=-1, verbose=1)

    # Fit the RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Best model from the random search
    best_rf = random_search.best_estimator_

    # Evaluate model performance
    y_pred = best_rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Show model performance metrics
    st.write(f"**Model Evaluation:**")
    st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.4f}")
    st.write(f"**R² (R-squared):** {r2:.4f}")

    # Feature columns for user input
    feature_cols = df.drop(columns=['Global_active_power']).columns
    user_input = []

    st.write("Enter Feature Values for Prediction")

    # Predefined values for ease of use (defaults)
    predefined_values = {
        'Global_intensity': 6.2,
        'Voltage': 240.0,
        'Sub_metering_1': 0.0,
        'Sub_metering_2': 1.0,
        'Sub_metering_3': 17.0,
        'Year': 2007,
        'Month': 12,
        'Day': 14,
        'Hour': 18,
        'Minute': 22,
        'Global_reactive_power': 0.2  # Added default value
    }

    # Gather user input
    for col in predefined_values:
        user_input.append(st.number_input(f"{col}", value=predefined_values[col]))

    # Check if user_input has the correct number of values (matching the feature columns)
    if len(user_input) < len(feature_cols):
        user_input.extend([None] * (len(feature_cols) - len(user_input)))  # Add None for missing values

    # Convert user_input into DataFrame
    input_df = pd.DataFrame([user_input], columns=feature_cols)

    # Encode the 'Weekday' column in the user input
    weekday = st.selectbox("Select the Weekday", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    input_df['Weekday'] = le.transform([weekday])[0]  # Transform the input weekday to match the encoding used in training

    # Make prediction with the best model
    prediction = best_rf.predict(input_df)[0]

    # Display the prediction
    st.write(f"Predicted Global Active Power: {prediction:.2f} kW")

    # Feature importance
    st.write("### Feature Importance")
    feature_importances = best_rf.feature_importances_

    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importances
    })

    # Sort the features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Display feature importance
    st.write(feature_importance_df)
