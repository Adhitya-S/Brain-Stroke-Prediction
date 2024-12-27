import streamlit as st
import pandas as pd
import joblib

# Load the model
stroke_model = joblib.load("model.joblib")

# Helper function for prediction
def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    encoded_cols, numeric_cols = stroke_model["encoded_cols"], stroke_model["numeric_cols"]
    preprocessor = stroke_model["preprocessor"]
    input_df[encoded_cols] = preprocessor.transform(input_df)
    X = input_df[numeric_cols + encoded_cols]
    prediction = stroke_model['model'].predict(X)
    return prediction

# Streamlit app
st.title("Brain Stroke Prediction Using ML")

# Add additional information using markdown
st.markdown("""
    <p>Welcome to the Brain Stroke Prediction website, where cutting-edge machine learning meets healthcare to predict the probability of a brain stroke. Advanced data analytics and machine learning algorithms are used to analyze critical health parameters such as age, blood pressure, cholesterol levels, heart rate, and lifestyle habits for people who may be at risk.</p>

    <p>Whether you need to evaluate your personal risk or wish to understand better about preventing a stroke, our platform provides easy-to-use access with personalized predictions that lead to your next step. Our usage of AI and data-driven models will help equip our users with early detection for the reduction of impact that may have otherwise been caused by a stroke while boosting overall well-being.</p>

    <p>Stroke is the second best leading cause of death globally, responsible for approximately 11% of total deaths (WHO). Stroke is a medical condition characterized by disrupted blood supply to the brain, leading to cellular death. Signs and symptoms of a stroke may include an inability to move or feel on one side of the body, problems understanding or speaking, dizziness, or loss of vision to one side.</p>

    <p>Take control of your health today, because a simple prediction may just change your tomorrow!</p>
""", unsafe_allow_html=True)

# Create a 2-column layout for the inputs
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"]).lower()
    age = st.number_input("Age", min_value=0, step=1)
    hypertension = st.selectbox("Hypertension", ["Yes", "No"]).lower()
    hypertension = 1 if hypertension == "yes" else 0
    ever_married = st.selectbox("Ever Married", ["Yes", "No"]).lower()
    work_type = st.selectbox("Work Type", ["Government job", "Children", "Never Worked", "Private"])

with col2:
    heart_disease = st.selectbox("Heart Disease", ["Yes", "No"]).lower()
    heart_disease = 1 if heart_disease == "yes" else 0
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    smoking_status = st.selectbox("Smoking Status", ["Unknown", "formerly smoked", "never smoked", "smokes"]).lower()

# Map work type
work_type_mapping = {
    "Government job": "Govt_job",
    "Children": "children",
    "Never Worked": "Never_worked",
    "Private": "Private",
}

# Prepare single input dictionary
single_input = {
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type_mapping.get(work_type, work_type),
    "Residence_type": residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status,
}

# Layout: Create two columns for buttons in each row
col3, col4 = st.columns(2)

with col3:
    if st.button("Predict"):
        prediction = predict_input(single_input)
        result = "The Person Has A Brain Stroke" if prediction[0] == 1 else "The Person Does Not Have A Brain Stroke"
        st.write(f"Prediction: {result}")

        # Show Help section based on prediction
        if result == "The Person Has A Brain Stroke":
            st.write("### Medication Suggestions:")
            st.write("""
                - **Aspirin:** Helps reduce the risk of stroke by thinning the blood.
                - **Blood Pressure Medications:** To control hypertension, which is a major stroke risk factor.
                - **Statins:** Used to lower cholesterol levels and reduce the risk of stroke.
                - **Anticoagulants:** To prevent blood clots that can lead to strokes.
                - **Diabetes Medications:** If you have diabetes, these can help control your blood sugar levels.
                - **Lifestyle Changes:** Regular exercise, maintaining a healthy diet, and quitting smoking are important.
            """)
        else:
            st.write("No medications are recommended since the stroke risk is low.")

with col4:
    if st.button("Clear"):
        # Add logic to clear inputs if needed
        st.write("Inputs cleared!")

# Help Section (always visible with medications)
col5, col6 = st.columns(2)

with col5:
    if st.button("Medications For Likely"):
        st.write("### Medication Suggestions:")
        st.write("""
            - **Aspirin:** Helps reduce the risk of stroke by thinning the blood.
            - **Blood Pressure Medications:** To control hypertension, which is a major stroke risk factor.
            - **Statins:** Used to lower cholesterol levels and reduce the risk of stroke.
            - **Anticoagulants:** To prevent blood clots that can lead to strokes.
            - **Diabetes Medications:** If you have diabetes, these can help control your blood sugar levels.
            - **Lifestyle Changes:** Regular exercise, maintaining a healthy diet, and quitting smoking are important.
        """)

with col6:
    if st.button("About"):
        # About logic or action
        st.write("The Stroke Prediction App uses machine learning to check the risk of having a nrain stroke based on key health factors such as age, gender, hypertension, and smoking habits. After feeding simple details, you are provided with a personalized risk prediction . In case your risk is high, the app gives you recommendations for certain medications and lifestyle changes in order to minimize the stroke risk. It acts as an informative tool for early detection and prevention of stroke. Always refer to a doctor for a personalized prescription and treatment.")
