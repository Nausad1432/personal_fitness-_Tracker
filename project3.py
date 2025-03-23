import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import warnings
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ---- PAGE CONFIGURATION ----
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

# Determine Streamlit's base background color
streamlit_background_color = "#f0f2f6"

# Custom CSS for styling
st.markdown(
    f"""
    <style>
    .main-header {{
        color: #262730;
        text-align: center;
        padding-bottom: 10px;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 30px;
    }}
    .user-input-header {{
        color: #4a5568;
        padding-top: 20px;
        margin-bottom: 15px;
    }}
    .stSlider label, .stRadio label {{
        color: #4a5568;
    }}
    .stDataFrame {{
        border: 1px solid #e2e8f0;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 20px;
    }}
    .prediction-header {{
        color: #1a73e8;
        margin-top: 30px;
        margin-bottom: 15px;
        text-align: center;
    }}
    .calories-burned {{
        font-size: 2.5em;
        font-weight: bold;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 20px;
    }}
    .similar-results-header {{
        color: #4a5568;
        margin-top: 30px;
        margin-bottom: 15px;
    }}
    .general-info-header {{
        color: #4a5568;
        margin-top: 30px;
        margin-bottom: 15px;
    }}
    .info-box {{
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        color: #2d3748;
    }}
    .success-message {{
        background-color: #f0fff4;
        border: 1px solid #b2f5c3;
        color: #276749;
        padding: 15px;
        border-radius: 5px;
        margin-top: 30px;
        text-align: center;
    }}
    .lottie-container {{ /* Style for centering and pseudo-transparency */
        display: flex;
        justify-content: center;
        background-color: {streamlit_background_color}; /* Match Streamlit's background */
        padding: 10px; /* Optional padding */
        border-radius: 5px; /* Optional rounded corners */
        margin-bottom: 20px; /* Add some space below the animation */
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- GLOBAL VARIABLES ----
ANIMATION_FILE = "Animation_1742061586293.json"
CALORIES_DATA_PATH = "calories.csv"
EXERCISE_DATA_PATH = "exercise.csv"
RANDOM_SEED = 42 # For reproducibility

# ---- HELPER FUNCTIONS ----
@st.cache_data
def load_data(calories_path, exercise_path):
    """Loads and merges the calories and exercise datasets."""
    try:
        calories = pd.read_csv(calories_path)
        exercise = pd.read_csv(exercise_path)
        exercise_df = exercise.merge(calories, on="User_ID").drop("User_ID", axis=1)
        logging.info("Data loaded and merged successfully.")
        return exercise_df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        logging.error(f"Error loading data: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocesses the exercise dataframe by adding BMI and converting gender."""
    if df is None:
        return None
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['BMI'] = round(df['BMI'], 2)
    df['Gender'] = df['Gender'].map({'female': 0, 'male': 1})
    logging.info("Data preprocessing completed.")
    return df

@st.cache_data
def split_data(df, test_size, random_state):
    """Splits the dataframe into training and testing sets."""
    if df is None:
        return None, None, None, None
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    X_train = train_data.drop("Calories", axis=1)
    y_train = train_data["Calories"]
    X_test = test_data["Calories"]
    y_test = test_data["Calories"]
    logging.info(f"Data split into training and testing sets (test size: {test_size}).")
    return X_train, y_train, X_test, y_test

@st.cache_resource
def train_model(X_train, y_train, n_estimators, max_features, max_depth, random_state):
    """Trains a RandomForestRegressor model."""
    model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    return model

# ---- PAGE CONTENT ----

# ---- HEADER SECTION ----
st.markdown("<h1 class='main-header'>🏋️ Personal Fitness Tracker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Track your calories burned with precise predictions based on your body metrics and exercise details.</p>", unsafe_allow_html=True)

# ---- SIDEBAR INPUT ----
st.sidebar.markdown("<h2 class='user-input-header'>User Input</h2>", unsafe_allow_html=True)

def user_input_features():
    """Collects user input features from the sidebar."""
    age = st.sidebar.slider("Age", 10, 100, 30, step=1)
    bmi = st.sidebar.slider("BMI", 15, 40, 20, step=1)
    duration = st.sidebar.slider("Duration (min)", 0, 120, 30, step=5)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 180, 100, step=5)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36.0, 42.0, 37.0, step=0.1)
    gender = 1 if st.sidebar.radio("Gender", ["Male", "Female"]) == "Male" else 0

    input_data = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender": [gender]
    })
    return input_data

df_user_input = user_input_features()

# ---- DATA LOADING AND PREPROCESSING ----
exercise_df = load_data(CALORIES_DATA_PATH, EXERCISE_DATA_PATH)

if exercise_df is not None:
    exercise_df = preprocess_data(exercise_df)

    # ---- DISPLAY USER INPUT ----
    st.markdown("---")
    st.subheader("Your Parameters")
    st.dataframe(df_user_input)

    # ---- DATA SPLITTING ----
    X_train, y_train, X_test, y_test = split_data(exercise_df, test_size=0.2, random_state=RANDOM_SEED)

    if X_train is not None:
        # ---- MODEL TRAINING ----
        model = train_model(X_train, y_train, n_estimators=500, max_features='sqrt', max_depth=10, random_state=RANDOM_SEED)

        # ---- PREDICTION ----
        if model:
            df_processed_input = df_user_input.reindex(columns=X_train.columns, fill_value=0)

            with st.spinner("⏳ Calculating your calories..."):
                time.sleep(1)
                prediction = model.predict(df_processed_input)
                predicted_calories = max(0, prediction[0])

            st.markdown("<h2 class='prediction-header'>🔥 Predicted Calories Burned</h2>", unsafe_allow_html=True)
            st.markdown(f"<p class='calories-burned'>{int(predicted_calories)} kcal</p>", unsafe_allow_html=True)

            # Sweating and Exhausted Animation (Centered with Attempt at Background Matching)
            try:
                with open(ANIMATION_FILE) as f:
                    animation_data = json.load(f)
                st.markdown(f"<div style='display: flex; justify-content: center; background-color: {streamlit_background_color}; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
                st_lottie(animation_data, speed=1, width=200, height=150)
                st.markdown("</div>", unsafe_allow_html=True)
            except FileNotFoundError:
                st.error(f"Error: Animation file '{ANIMATION_FILE}' not found.")
                logging.error(f"Error: Animation file '{ANIMATION_FILE}' not found.")
            except json.JSONDecodeError:
                st.error(f"Error: Could not decode JSON from '{ANIMATION_FILE}'.")
                logging.error(f"Error: Could not decode JSON from '{ANIMATION_FILE}'.")

            # ---- CALORIE DISTRIBUTION CHART ----
            st.markdown("<h2 class='animated-chart-header'>📈 Calorie Distribution</h2>", unsafe_allow_html=True)
            calorie_chart = alt.Chart(exercise_df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=alt.X('Calories', bin=alt.Bin(maxbins=30), title="Calories Burned"),
                y=alt.Y('count()', title="Number of Users"),
                tooltip=['Calories', 'count()']
            ).properties(
                width=700,
                height=400,
                title="Distribution of Calories Burned"
            ).interactive()
            st.altair_chart(calorie_chart, use_container_width=True)

            # ---- GENERAL INFORMATION ----
            st.markdown("<h2 class='general-info-header'>ℹ️ General Information</h2>", unsafe_allow_html=True)

            def calculate_percentage_higher(column_name, user_value, df):
                """Calculates the percentage of users with a lower value in a given column."""
                if df is not None and column_name in df.columns:
                    lower_count = sum(df[column_name] < user_value)
                    total_count = len(df)
                    if total_count > 0:
                        percentage = round((lower_count / total_count) * 100, 2)
                        return f"Your {column_name.lower().replace('_', ' ')} is higher than *{percentage}%* of other users."
                    else:
                        return "Not enough data to compare."
                else:
                    return f"Column '{column_name}' not found in the dataset."

            st.markdown(f"<div class='info-box'>{calculate_percentage_higher('Age', df_user_input['Age'].values[0], exercise_df)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-box'>{calculate_percentage_higher('Duration', df_user_input['Duration'].values[0], exercise_df)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-box'>{calculate_percentage_higher('Heart_Rate', df_user_input['Heart_Rate'].values[0], exercise_df)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-box'>{calculate_percentage_higher('Body_Temp', df_user_input['Body_Temp'].values[0], exercise_df)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-box'>{calculate_percentage_higher('BMI', df_user_input['BMI'].values[0], exercise_df)}</div>", unsafe_allow_html=True)

            st.markdown("<p class='success-message'>✅ Enjoy your healthy journey with the Personal Fitness Tracker!</p>", unsafe_allow_html=True)

        else:
            st.error("Model training failed. Please check the logs for more information.")

else:
    st.error("Failed to load or preprocess data. Please ensure the data files are in the correct location.")

# ---- FOOTER SECTION ----
footer = """
<style>
    html, body {
        height: 100%;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
    }

    .content {
        flex: 1; /* Ensures content expands to fill available space */
    }

    .footer {
        background-color: #2d3748;
        color: white;
        text-align: center;
        padding: 20px 0;
        font-size: 14px;
        width: 100%;
    }

    .footer a {
        color: #63b3ed;
        text-decoration: none;
    }

    .footer a:hover {
        text-decoration: underline;
    }
</style>

<div class="content">
    <!-- Main content goes here -->
</div>

<div class="footer">
    <p>🧑‍💻 Created with ❤️ by Manjot Singh | <a href="https://www.yourwebsite.com" target="_blank">Personal Fitness Tracker</a></p>
    <p>📞 Contact: Manjotmattu78922@gmail.com | 📍 Location: Ludhiana, Punjab</p>
    <p>&copy; 2025 Personal Fitness Tracker. All Rights Reserved.</p>
</div>

"""
st.markdown(footer, unsafe_allow_html=True)