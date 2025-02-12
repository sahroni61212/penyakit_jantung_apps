#1. Read Library
import streamlit as st #untuk pembuatan web
import pandas as pd #baca data
import pickle #baca model
from sklearn.preprocessing import LabelEncoder #labeling input

#2. Load the saved model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

#3. Preprocessing Model
# Function to preprocess input data
def preprocess_input(data):
    # Convert categorical variables to numerical using LabelEncoder
    le_sex = LabelEncoder()
    le_cp = LabelEncoder()
    le_fbs = LabelEncoder()
    le_restecg = LabelEncoder()
    le_exang = LabelEncoder()
    le_slope = LabelEncoder()
    le_ca = LabelEncoder()
    le_thal = LabelEncoder()
    data['sex'] = le_sex.fit_transform(data['sex'])
    data['cp'] = le_cp.fit_transform(data['cp'])
    data['fbs'] = le_fbs.fit_transform(data['fbs'])
    data['restecg'] = le_restecg.fit_transform(data['restecg'])
    data['exang'] = le_exang.fit_transform(data['exang'])
    data['slope'] = le_slope.fit_transform(data['slope'])
    data['ca'] = le_ca.fit_transform(data['ca'])
    data['thal'] = le_thal.fit_transform(data['thal'])
    return data 


#4. Configuration of Streamlit
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#5. . Configuration of Background
st.markdown(
    """
    <style>
    .main {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #d9534f;
        text-align: center;
        font-family: 'Arial Black', sans-serif;
    }
    h2 {
        color: #333;
        font-family: 'Arial Black', sans-serif;
    }
    .stButton button {
        background-color: #d9534f;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #a83c3a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#6. Read File Image for Web
st.title("❤️Heart Disease Prediction❤️")
st.image("heart-disease.jpg",caption="Heart Disease Awareness", use_column_width=True)

#7. Iput from user

# Input fields for user
st.header("📝 Enter Patient Information")
col1, col2 = st.columns(2)  # Split into two columns for better layout
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymtomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
with col2:
    restecg = st.selectbox("Resting ECG Results", ["normal", "ST-T Wave abnormal", "probable or definite left ventricular hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
    ca = st.selectbox("Number of Major Vessels", ["Number of major vessels: 0", "Number of major vessels: 1", "Number of major vessels: 2", "Number of major vessels: 3"])
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

#8. Create Dataframe from User
input_data = pd.DataFrame({
    'age':[age],
    'sex':[sex],
    'cp':[cp],
    'trestbps':[trestbps],
    'chol':[chol],
    'fbs':[fbs],
    'restecg':[restecg],
    'thalach':[thalach],
    'exang':[exang],
    'oldpeak':[oldpeak],
    'slope':[slope],
    'ca':[ca],
    'thal':[thal]

})


#9. Preprocessing from Data Input
preprocessed_data=preprocess_input(input_data)
# 10. Get Result from Model and Input Data
if st.button("Predict"):
    prediction=model.predict(preprocessed_data)
    prediction_proba=model.predict_proba(preprocessed_data)
    #Display Result
    st.subheader("Prediction Result")
    if prediction[0]==0:
        st.succes("🎉 No Disease Detected!🎉")
    else:
        st.error("⚠️Disease Detected!⚠️")
    st.subheader("Prediction Probability")
    st.write(f"Probability of No Disease:{prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Disease:{prediction_proba[0][1]:.2f}")