import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved models
models = {
    "Decision Tree": joblib.load("decision_tree_model.joblib"),
    "Random Forest": joblib.load("random_forest_model.joblib"),
    "Gradient Boosting": joblib.load("gradient_boosting_model.joblib")
}

# Load the training column names
training_columns_file = "training_columns.joblib"
training_columns = joblib.load(training_columns_file)


# Define the preprocessing function
def preprocess_input(user_input, training_columns):
    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_input])

    # Identify categorical columns from the original df (excluding 'pass' and the target)
    # These were identified in the notebook
    cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                'nursery', 'higher', 'internet', 'romantic']

    # Create an empty DataFrame with the correct columns and data types based on training data
    processed_input = pd.DataFrame(columns=training_columns)

    # Fill the DataFrame with default values (0 for int/float, False for bool)
    for col in training_columns:
        # Infer data type from column name (simplified)
        # A more robust approach would be to save and load data types from training data
        if any(substring in col for substring in ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                                                 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                                                 'absences', 'G1', 'G2', 'G3']):
             processed_input[col] = 0
        else: # These are the boolean columns from OHE
             processed_input[col] = False


    # Update the DataFrame with user input
    for key, value in user_input.items():
        if key in processed_input.columns:
            processed_input[key] = value
        elif isinstance(value, str) and f"{key}_{value}" in processed_input.columns:
             # Handle encoded categorical features
             # This assumes the format 'columnname_category' matches the OHE output
             processed_input[f"{key}_{value}"] = True
        elif isinstance(value, bool) and f"{key}_yes" in processed_input.columns:
             # Handle binary categorical features like schoolsup, famsup, etc.
             if value:
                 processed_input[f"{key}_yes"] = True
        elif isinstance(value, str) and f"{key}_{value.lower()}" in processed_input.columns:
             # Handle case variations in user input for categorical features
             processed_input[f"{key}_{value.lower()}"] = True


    # Ensure the order of columns matches the training data
    processed_input = processed_input[training_columns]

    # Convert boolean columns to int (0 or 1) - this was identified as a potential issue
    processed_input = processed_input.astype(int)


    return processed_input


# Streamlit App Title
st.title("🎓 Student Performance Predictor")

st.write("Enter student details to predict if they will pass (>= 10) or fail (< 10).")

# --- Input Widgets ---
st.header("Student Information")

# Numerical Inputs
age = st.number_input("Age", min_value=15, max_value=22, value=17)
medu = st.selectbox("Mother's Education (0-4)", options=[0, 1, 2, 3, 4])
fedu = st.selectbox("Father's Education (0-4)", options=[0, 1, 2, 3, 4])
traveltime = st.selectbox("Travel Time (1-4)", options=[1, 2, 3, 4])
studytime = st.selectbox("Study Time (1-4)", options=[1, 2, 3, 4])
failures = st.number_input("Past Failures", min_value=0, max_value=3, value=0)
famrel = st.selectbox("Family Relationship (1-5)", options=[1, 2, 3, 4, 5])
freetime = st.selectbox("Free Time (1-5)", options=[1, 2, 3, 4, 5])
goout = st.selectbox("Going Out (1-5)", options=[1, 2, 3, 4, 5])
dalc = st.selectbox("Workday Alcohol (1-5)", options=[1, 2, 3, 4, 5])
walc = st.selectbox("Weekend Alcohol (1-5)", options=[1, 2, 3, 4, 5])
health = st.selectbox("Health Status (1-5)", options=[1, 2, 3, 4, 5])
absences = st.number_input("Absences", min_value=0, value=0)
g1 = st.number_input("First Period Grade (G1)", min_value=0, max_value=20, value=10)
g2 = st.number_input("Second Period Grade (G2)", min_value=0, max_value=20, value=10)
g3 = st.number_input("Final Grade (G3)", min_value=0, max_value=20, value=10) # Include G3 as it's a strong predictor, though might lead to perfect scores if used directly


# Categorical Inputs (using selectbox for simplicity)
school = st.selectbox("School", options=['GP', 'MS'])
sex = st.selectbox("Sex", options=['F', 'M'])
address = st.selectbox("Address Type", options=['U', 'R'])
famsize = st.selectbox("Family Size", options=['GT3', 'LE3'])
pstatus = st.selectbox("Parent's Cohabitation Status", options=['T', 'A'])
mjob = st.selectbox("Mother's Job", options=['teacher', 'health', 'services', 'at_home', 'other'])
fjob = st.selectbox("Father's Job", options=['teacher', 'health', 'services', 'at_home', 'other'])
reason = st.selectbox("Reason for school choice", options=['home', 'reputation', 'course', 'other'])
guardian = st.selectbox("Guardian", options=['mother', 'father', 'other'])
schoolsup = st.selectbox("School Support", options=['yes', 'no'])
famsup = st.selectbox("Family Support", options=['yes', 'no'])
paid = st.selectbox("Paid Classes", options=['yes', 'no'])
activities = st.selectbox("Extracurricular Activities", options=['yes', 'no'])
nursery = st.selectbox("Attended Nursery School", options=['yes', 'no'])
higher = st.selectbox("Desire for Higher Education", options=['yes', 'no'])
internet = st.selectbox("Internet Access", options=['yes', 'no'])
romantic = st.selectbox("In a Romantic Relationship", options=['yes', 'no'])


# --- Prediction Button ---
if st.button("Predict Pass/Fail"):
    # Collect user input into a dictionary
    user_input = {
        'age': age,
        'Medu': medu,
        'Fedu': fedu,
        'traveltime': traveltime,
        'studytime': studytime,
        'failures': failures,
        'famrel': famrel,
        'freetime': freetime,
        'goout': goout,
        'Dalc': dalc,
        'Walc': walc,
        'health': health,
        'absences': absences,
        'G1': g1,
        'G2': g2,
        'G3': g3, # Include G3
        'school': school,
        'sex': sex,
        'address': address,
        'famsize': famsize,
        'Pstatus': pstatus,
        'Mjob': mjob,
        'Fjob': fjob,
        'reason': reason,
        'guardian': guardian,
        'schoolsup': schoolsup,
        'famsup': famsup,
        'paid': paid,
        'activities': activities,
        'nursery': nursery,
        'higher': higher,
        'internet': internet,
        'romantic': romantic
    }

    # Preprocess the input using the loaded training columns
    processed_input_df = preprocess_input(user_input, training_columns)

    st.subheader("Prediction Results:")

    # Make predictions with each model
    for name, model in models.items():
        prediction = model.predict(processed_input_df)[0]
        if prediction == 1:
            st.success(f"{name}: Pass")
        else:
            st.error(f"{name}: Fail")
