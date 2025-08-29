import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Load models safely
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Set paths (use your actual model paths or relative paths)
model_dir = 'C:/Users/Durvank/Downloads/ML Project/Models/'
diabetes_model = load_model(os.path.join(model_dir, 'diabetes_model.sav'))
heart_disease_model = load_model(os.path.join(model_dir, 'heart_disease_model.sav'))
parkinsons_model = load_model(os.path.join(model_dir, 'parkinsons_model.sav'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System by Durvank',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# Helper: show prediction & optional confidence
def show_prediction(model, user_input, pos_label, neg_label):
    prediction = model.predict([user_input])
    diagnosis = pos_label if prediction[0] == 1 else neg_label
    st.success(diagnosis)

    # Try to get probability if supported
    if hasattr(model, "predict_proba"):
        try:
            prediction_prob = model.predict_proba([user_input])[0][1]  # Probability of positive class
            st.info(f"Confidence Score: {prediction_prob * 100:.2f}%")
        except Exception:
            pass

# ---------------------- DIABETES PAGE ----------------------
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1)
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0, max_value=300)
    with col3:
        BloodPressure = st.number_input('Blood Pressure', min_value=0, max_value=200)
    with col1:
        SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=100)
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0, max_value=1000)
    with col3:
        BMI = st.number_input('BMI', min_value=0.0, max_value=70.0, format="%.1f")
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, format="%.2f")
    with col2:
        Age = st.number_input('Age', min_value=0, max_value=120)

    if st.button('Diabetes Test Result') and diabetes_model:
        try:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                          BMI, DiabetesPedigreeFunction, Age]
            show_prediction(diabetes_model, user_input,
                            'The person is diabetic',
                            'The person is not diabetic')
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------- HEART DISEASE PAGE ----------------------
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120)
    with col2:
        sex = st.number_input('Sex (1=Male, 0=Female)', min_value=0, max_value=1)
    with col3:
        cp = st.number_input('Chest Pain Type (0–3)', min_value=0, max_value=3)
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=200)
    with col2:
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=0, max_value=600)
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 (1=True, 0=False)', min_value=0, max_value=1)
    with col1:
        restecg = st.number_input('Resting ECG Result (0–2)', min_value=0, max_value=2)
    with col2:
        thalach = st.number_input('Max Heart Rate', min_value=0, max_value=300)
    with col3:
        exang = st.number_input('Exercise Induced Angina (1=True, 0=False)', min_value=0, max_value=1)
    with col1:
        oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=10.0, format="%.1f")
    with col2:
        slope = st.number_input('Slope (0–2)', min_value=0, max_value=2)
    with col3:
        ca = st.number_input('Major Vessels Colored (0–3)', min_value=0, max_value=3)
    with col1:
        thal = st.number_input('Thal (0=Normal, 1=Fixed, 2=Reversible)', min_value=0, max_value=2)

    if st.button('Heart Disease Test Result') and heart_disease_model:
        try:
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                          exang, oldpeak, slope, ca, thal]
            show_prediction(heart_disease_model, user_input,
                            'The person is having heart disease',
                            'The person does not have heart disease')
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------- PARKINSON'S PAGE ----------------------
elif selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    col1, col2, col3, col4, col5 = st.columns(5)

    def num_input(label):
        return st.number_input(label, format="%.4f")

    with col1:
        fo = num_input('MDVP:Fo(Hz)')
    with col2:
        fhi = num_input('MDVP:Fhi(Hz)')
    with col3:
        flo = num_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = num_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = num_input('MDVP:Jitter(Abs)')
    with col1:
        RAP = num_input('MDVP:RAP')
    with col2:
        PPQ = num_input('MDVP:PPQ')
    with col3:
        DDP = num_input('Jitter:DDP')
    with col4:
        Shimmer = num_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = num_input('MDVP:Shimmer(dB)')
    with col1:
        APQ3 = num_input('Shimmer:APQ3')
    with col2:
        APQ5 = num_input('Shimmer:APQ5')
    with col3:
        APQ = num_input('MDVP:APQ')
    with col4:
        DDA = num_input('Shimmer:DDA')
    with col5:
        NHR = num_input('NHR')
    with col1:
        HNR = num_input('HNR')
    with col2:
        RPDE = num_input('RPDE')
    with col3:
        DFA = num_input('DFA')
    with col4:
        spread1 = num_input('spread1')
    with col5:
        spread2 = num_input('spread2')
    with col1:
        D2 = num_input('D2')
    with col2:
        PPE = num_input('PPE')

    if st.button("Parkinson's Test Result") and parkinsons_model:
        try:
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                          Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                          RPDE, DFA, spread1, spread2, D2, PPE]
            show_prediction(parkinsons_model, user_input,
                            "The person has Parkinson's disease",
                            "The person does not have Parkinson's disease")
        except Exception as e:
            st.error(f"Prediction error: {e}")
