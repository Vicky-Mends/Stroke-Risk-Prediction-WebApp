import os
import joblib
import streamlit as st
import numpy as np

# ── Polynomial feature helper ─────────────────────────────────────────────────
def add_poly(X):
    age       = X[:, 0]
    glu       = X[:, 1]
    age_sq    = age ** 2
    inter     = age * glu
    glu_sq    = glu ** 2
    return np.c_[X, age_sq, inter, glu_sq]

# ── Scaler parameters from training ────────────────────────────────────────────
SCALER_MEAN  = np.array([47.4572, 106.1478, 0.0482, 0.0513, 0.5527,
                         0.5431,   2.1356,   0.5064, 1850.37, 5067.84, 11645.2])
SCALER_SCALE = np.array([15.6753,  26.8145, 0.2141, 0.2206, 0.4974,
                         0.4983,   0.9082,   0.4999, 2978.41, 6144.78, 10795.6])

# ── Load bare model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(os.path.join(base, "best_gb_model.pkl"))

model = load_model()

# ── Page config & CSS ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Stroke Risk Assessment", layout="wide")
st.markdown("""
    <style>
      #MainMenu, footer, header {visibility: hidden;}
      [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# ── Title & Navbar ─────────────────────────────────────────────────────────────
st.title("📝 Stroke Risk Assessment")
st.markdown("""
  <style>
    .custom-nav {
      background: #e8f5e9; padding: 15px 0; border-radius: 10px;
      display: flex; justify-content: center; gap: 60px; margin-bottom: 30px;
      font-size: 18px; font-weight: 600;
    }
    .custom-nav a { text-decoration: none; color: #4C9D70; }
    .custom-nav a:hover { color: #388e3c; text-decoration: underline; }
  </style>
  <div class="custom-nav">
    <a href='/Home'>Home</a>
    <a href='/Risk_Assessment'>Risk Assessment</a>
    <a href='/Results'>Results</a>
    <a href='/Recommendations'>Recommendations</a>
  </div>
""", unsafe_allow_html=True)

# ── Input Sections ─────────────────────────────────────────────────────────────
with st.expander("👤 Personal Information", expanded=True):
    age = st.number_input("Age",    min_value=18, max_value=100, value=18, step=1, key="age")
    gender = st.selectbox("Gender", ["Select option", "Male", "Female"], key="gender")
    ever_married = st.selectbox("Ever Married?", ["Select option", "Yes", "No"], key="ever_married")
    work_type = st.selectbox("Work Type", ["Select option", "Private", "Self-employed", "Govt_job", "Never_worked"], key="work_type")

with st.expander("🩺 Health Information", expanded=True):
    hypertension = st.radio("Do you have hypertension?", ["Select option", "Yes", "No"], key="hypertension")
    heart_disease = st.radio("Do you have heart disease?", ["Select option", "Yes", "No"], key="heart_disease")
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=55.0, max_value=300.0, value=55.0, step=0.1, key="avg_glucose_level")
    smoking_status = st.selectbox("Smoking Status", ["Select option", "never smoked", "formerly smoked", "smokes"], key="smoking_status")

# ── Consent & Disclaimer ───────────────────────────────────────────────────────
st.markdown("### 📄 Consent and Disclaimer")
st.write(
    "This tool provides an estimate of stroke risk based on the information you provide. "
    "It is not a diagnostic tool and should not replace professional medical advice. "
    "By submitting, you agree to allow us to estimate your stroke risk."
)
st.checkbox("I agree to the terms and allow risk estimation", key="consent")

# ── Calculate & Redirect ────────────────────────────────────────────────────────
if st.button("Calculate Stroke Risk 📈"):
    if not st.session_state.consent:
        st.error("You must agree to the terms before proceeding!")
    elif any(val == "Select option" for val in [gender, ever_married, work_type, hypertension, heart_disease, smoking_status]):
        st.error("Please complete all fields with valid values before submitting.")
    else:
        # build raw feature vector in training order
        heart_map = {"Yes":1, "No":0}
        htn_map   = {"Yes":1, "No":0}
        married_map = {"Yes":1, "No":0}
        smoke_map = {"never smoked":0, "formerly smoked":1, "smokes":2}
        work_map  = {"Private":0, "Self-employed":1, "Govt_job":2, "Never_worked":3}
        gender_map= {"Male":0, "Female":1}

        raw = [
            age,
            avg_glucose_level,
            heart_map[heart_disease],
            htn_map[hypertension],
            married_map[ever_married],
            smoke_map[smoking_status],
            work_map[work_type],
            gender_map[gender]
        ]
        X_raw = np.array(raw).reshape(1, -1)
        X_poly= add_poly(X_raw)
        X_scaled = (X_poly - SCALER_MEAN) / SCALER_SCALE
        prob = model.predict_proba(X_scaled)[0, 1]

        # save session
        st.session_state.user_data = {
            "age": age,
            "avg_glucose_level": avg_glucose_level,
            "heart_disease": heart_disease,
            "hypertension": hypertension,
            "ever_married": ever_married,
            "smoking_status": smoking_status,
            "work_type": work_type,
            "gender": gender
        }
        st.session_state.prediction_prob = prob
        st.switch_page("pages/Results.py")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
  <style>
    .custom-footer { background-color: rgba(76,157,112,0.6); color: white; padding: 30px 0; border-radius: 12px; margin-top: 40px; text-align: center; font-size: 14px; width: 100%; }
    .custom-footer a { color: white; text-decoration: none; margin: 0 15px; }
    .custom-footer a:hover { text-decoration: underline; }
  </style>
  <div class="custom-footer">
      <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
      <p><a href='/Home'>Home</a> <a href='/Risk_Assessment'>Risk Assessment</a> <a href='/Results'>Results</a> <a href='/Recommendations'>Recommendations</a></p>
      <p style="font-size:12px;">Developed by Victoria Mends</p>
  </div>
""", unsafe_allow_html=True)








# import os
# import joblib
# import streamlit as st
# import numpy as np
# import sys, __main__

# # ── Polynomial helper for pipeline unpickling ─────────────────────────────────
# def add_poly(X_array):
#     age         = X_array[:, 0]
#     glu         = X_array[:, 1]
#     age_sq      = age ** 2
#     interaction = age * glu
#     glu_sq      = glu ** 2
#     return np.c_[X_array, age_sq, interaction, glu_sq]

# # Expose to __main__ for unpickling
# setattr(__main__, "add_poly", add_poly)

# # ── Load end-to-end pipeline ─────────────────────────────────────────────────
# @st.cache_resource
# def load_pipeline():
#     base = os.path.dirname(os.path.abspath(__file__))
#     return joblib.load(os.path.join(base, "best_gb_pipeline.pkl"))

# pipeline = load_pipeline()

# # ── Page config & CSS ─────────────────────────────────────────────────────────
# st.set_page_config(page_title="Stroke Risk Assessment", layout="wide")
# st.markdown("""
#     <style>
#       #MainMenu, footer, header {visibility: hidden;}
#       [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
#     </style>
# """, unsafe_allow_html=True)

# # ── Title & Navbar ─────────────────────────────────────────────────────────────
# st.title("📝 Stroke Risk Assessment")
# st.markdown("""
#   <style>
#     .custom-nav {
#       background: #e8f5e9; padding: 15px 0; border-radius: 10px;
#       display: flex; justify-content: center; gap: 60px; margin-bottom: 30px;
#       font-size: 18px; font-weight: 600;
#     }
#     .custom-nav a { text-decoration: none; color: #4C9D70; }
#     .custom-nav a:hover { color: #388e3c; text-decoration: underline; }
#   </style>
#   <div class="custom-nav">
#     <a href='/Home'>Home</a>
#     <a href='/Risk_Assessment'>Risk Assessment</a>
#     <a href='/Results'>Results</a>
#     <a href='/Recommendations'>Recommendations</a>
#   </div>
# """, unsafe_allow_html=True)

# # ── Input Sections ─────────────────────────────────────────────────────────────
# with st.expander("👤 Personal Information", expanded=True):
#     age = st.number_input("Age", min_value=18, max_value=100, value=18, step=1, key="age")
#     gender = st.selectbox("Gender", ["Select option", "Male", "Female"], key="gender")
#     ever_married = st.selectbox("Ever Married?", ["Select option", "Yes", "No"], key="ever_married")
#     work_type = st.selectbox("Work Type", ["Select option", "Private", "Self-employed", "Govt_job", "Never_worked"], key="work_type")

# with st.expander("🩺 Health Information", expanded=True):
#     hypertension = st.radio("Do you have hypertension?", ["Select option", "Yes", "No"], key="hypertension")
#     heart_disease = st.radio("Do you have heart disease?", ["Select option", "Yes", "No"], key="heart_disease")
#     avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=55.0, max_value=300.0, value=55.0, step=0.1, key="avg_glucose_level")
#     smoking_status = st.selectbox("Smoking Status", ["Select option", "never smoked", "formerly smoked", "smokes"], key="smoking_status")

# # ── Consent & Disclaimer ───────────────────────────────────────────────────────
# st.markdown("### 📄 Consent and Disclaimer")
# st.write(
#     "This tool provides an estimate of stroke risk based on the information you provide. "
#     "It is not a diagnostic tool and should not replace professional medical advice. "
#     "By submitting, you agree to allow us to estimate your stroke risk."
# )
# st.checkbox("I agree to the terms and allow risk estimation", key="consent")

# # ── Calculate & Redirect ────────────────────────────────────────────────────────
# if st.button("Calculate Stroke Risk 📈"):
#     if not st.session_state.consent:
#         st.error("You must agree to the terms before proceeding!")
#     elif any(val == "Select option" for val in [gender, ever_married, work_type, hypertension, heart_disease, smoking_status]):
#         st.error("Please complete all fields with valid values before submitting.")
#     else:
#         # build raw feature vector in training order
#         raw = [
#             age,
#             avg_glucose_level,
#             1 if heart_disease == "Yes" else 0,
#             1 if hypertension   == "Yes" else 0,
#             1 if ever_married    == "Yes" else 0,
#             {"never smoked":0, "formerly smoked":1, "smokes":2}[smoking_status],
#             {"Private":0, "Self-employed":1, "Govt_job":2, "Never_worked":3}[work_type],
#             {"Male":0, "Female":1}[gender]
#         ]
#         features = np.array(raw).reshape(1, -1)
#         prob = pipeline.predict_proba(features)[0, 1]

#         # save to session
#         st.session_state.user_data = {
#             "age": age,
#             "avg_glucose_level": avg_glucose_level,
#             "heart_disease": heart_disease,
#             "hypertension": hypertension,
#             "ever_married": ever_married,
#             "smoking_status": smoking_status,
#             "work_type": work_type,
#             "gender": gender
#         }
#         st.session_state.prediction_prob = prob
#         st.switch_page("pages/Results.py")

# # ── Footer ────────────────────────────────────────────────────────────────────
# st.markdown("""
#   <style>
#     .custom-footer { background-color: rgba(76,157,112,0.6); color: white; padding: 30px 0; border-radius: 12px; margin-top: 40px; text-align: center; font-size: 14px; width: 100%; }
#     .custom-footer a { color: white; text-decoration: none; margin: 0 15px; }
#     .custom-footer a:hover { text-decoration: underline; }
#   </style>
#   <div class="custom-footer">
#       <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
#       <p><a href='/Home'>Home</a> <a href='/Risk_Assessment'>Risk Assessment</a> <a href='/Results'>Results</a> <a href='/Recommendations'>Recommendations</a></p>
#       <p style="font-size:12px;">Developed by Victoria Mends</p>
#   </div>
# """, unsafe_allow_html=True)






# import os
# import joblib
# import streamlit as st
# import numpy as np

# # ── Polynomial helper ─────────────────────────────────────────────────────────
# def add_poly(X_array):
#     age         = X_array[:, 0]
#     glu         = X_array[:, 1]
#     age_sq      = age ** 2
#     interaction = age * glu
#     glu_sq      = glu ** 2
#     return np.c_[X_array, age_sq, interaction, glu_sq]

# # ── Scaler parameters from training ───────────────────────────────────────────
# SCALER_MEAN  = np.array([47.4572, 106.1478,  0.0482, 0.0513, 0.5527,
#                          0.5431,   2.1356,   0.5064, 1850.37, 5067.84, 11645.2])
# SCALER_SCALE = np.array([15.6753,  26.8145,  0.2141, 0.2206, 0.4974,
#                          0.4983,   0.9082,   0.4999, 2978.41, 6144.78, 10795.6])

# # ── Page config & CSS ─────────────────────────────────────────────────────────
# st.set_page_config(page_title="Stroke Risk Assessment", layout="wide")
# st.markdown("""
#     <style>
#       #MainMenu, footer, header {visibility: hidden;}
#       [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
#     </style>
# """, unsafe_allow_html=True)

# # ── Title & Navbar ─────────────────────────────────────────────────────────────
# st.title("📝 Stroke Risk Assessment")
# st.markdown("""
#   <style>
#     .custom-nav {
#       background: #e8f5e9; padding: 15px 0; border-radius: 10px;
#       display: flex; justify-content: center; gap: 60px; margin-bottom: 30px;
#       font-size: 18px; font-weight: 600;
#     }
#     .custom-nav a { text-decoration: none; color: #4C9D70; }
#     .custom-nav a:hover { color: #388e3c; text-decoration: underline; }
#   </style>
#   <div class="custom-nav">
#     <a href='/Home'>Home</a>
#     <a href='/Risk_Assessment'>Risk Assessment</a>
#     <a href='/Results'>Results</a>
#     <a href='/Recommendations'>Recommendations</a>
#   </div>
# """, unsafe_allow_html=True)

# # ── Load bare model ────────────────────────────────────────────────────────────
# @st.cache_resource
# def load_model():
#     base = os.path.dirname(os.path.abspath(__file__))
#     return joblib.load(os.path.join(base, "best_gb_model.pkl"))

# model = load_model()

# # ── Input Sections ─────────────────────────────────────────────────────────────
# with st.expander("👤 Personal Information", expanded=True):
#     age = st.number_input("Age", 18, 100, value=18, step=1, key="age")
#     gender = st.selectbox("Gender", ["Select option", "Male", "Female"], key="gender")
#     ever_married = st.selectbox("Ever Married?", ["Select option", "Yes", "No"], key="ever_married")
#     work_type = st.selectbox("Work Type", ["Select option", "Private", "Self-employed", "Govt_job", "Never_worked"], key="work_type")

# with st.expander("🩺 Health Information", expanded=True):
#     hypertension = st.radio("Do you have hypertension?", ["Select option", "Yes", "No"], key="hypertension")
#     heart_disease = st.radio("Do you have heart disease?", ["Select option", "Yes", "No"], key="heart_disease")
#     avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", 55.0, 300.0, value=55.0, step=0.1, key="avg_glucose_level")
#     smoking_status = st.selectbox("Smoking Status", ["Select option", "never smoked", "formerly smoked", "smokes"], key="smoking_status")

# # ── Consent ────────────────────────────────────────────────────────────────────
# st.markdown("### 📄 Consent and Disclaimer")
# st.write("This tool provides an estimate of stroke risk. Not a diagnostic. Consult a professional.")
# st.checkbox("I agree to the terms and allow risk estimation", key="consent")

# # ── Calculate & Redirect ────────────────────────────────────────────────────────
# if st.button("Calculate Stroke Risk 📈"):
#     if not st.session_state.consent:
#         st.error("You must agree before proceeding!")
#     elif any(val == "Select option" for val in [gender, ever_married, work_type, hypertension, heart_disease, smoking_status]):
#         st.error("Please complete all fields.")
#     else:
#         # raw features order
#         raw = [
#             age,
#             avg_glucose_level,
#             1 if heart_disease == "Yes" else 0,
#             1 if hypertension   == "Yes" else 0,
#             1 if ever_married    == "Yes" else 0,
#             {"never smoked":0, "formerly smoked":1, "smokes":2}[smoking_status],
#             {"Private":0, "Self-employed":1, "Govt_job":2, "Never_worked":4}[work_type],
#             0 if gender == "Male" else 1
#         ]
#         X_raw = np.array(raw).reshape(1, -1)
#         X_poly = add_poly(X_raw)
#         X_scaled = (X_poly - SCALER_MEAN) / SCALER_SCALE
#         prob = model.predict_proba(X_scaled)[0, 1]

#         st.session_state.user_data       = {k: v for k, v in zip(["age","avg_glucose_level","heart_disease","hypertension","ever_married","smoking_status","work_type","gender"], raw)}
#         st.session_state.prediction_prob = prob
#         st.switch_page("pages/Results.py")

# # ── Footer ────────────────────────────────────────────────────────────────────
# st.markdown("""
#   <style>
#     .custom-footer { background-color: rgba(76,157,112,0.6); color: white; padding: 30px 0; border-radius: 12px; margin-top: 40px; text-align: center; font-size: 14px; width: 100%; }
#     .custom-footer a { color: white; text-decoration: none; margin: 0 15px; }
#     .custom-footer a:hover { text-decoration: underline; }
#   </style>
#   <div class="custom-footer">
#       <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
#       <p><a href='/Home'>Home</a> <a href='/Risk_Assessment'>Risk Assessment</a> <a href='/Results'>Results</a> <a href='/Recommendations'>Recommendations</a></p>
#       <p style="font-size:12px;">Developed by Victoria Mends</p>
#   </div>
# """, unsafe_allow_html=True)

















# import os
# import sys
# import __main__
# import joblib
# import streamlit as st
# import numpy as np

# # ── Define polynomial helper for unpickling ─────────────────────────────────
# def add_poly(X_array):
#     age         = X_array[:, 0]
#     glu         = X_array[:, 1]
#     age_sq      = age ** 2
#     interaction = age * glu
#     glu_sq      = glu ** 2
#     return np.c_[X_array, age_sq, interaction, glu_sq]

# # Monkey-patch into __main__ so pipeline can unpickle
# setattr(__main__, "add_poly", add_poly)
# def add_poly(X_array):
#     age         = X_array[:, 0]
#     glu         = X_array[:, 1]
#     age_sq      = age ** 2
#     interaction = age * glu
#     glu_sq      = glu ** 2
#     return np.c_[X_array, age_sq, interaction, glu_sq]

# # ── Page config & hide defaults ────────────────────────────────────────────────
# st.set_page_config(page_title="Stroke Risk Assessment", layout="wide")
# st.markdown("""
#     <style>
#       #MainMenu, footer, header {visibility: hidden;}
#       [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
#     </style>
# """, unsafe_allow_html=True)

# # ── Title & Navbar ─────────────────────────────────────────────────────────────
# st.title("📝 Stroke Risk Assessment")
# st.markdown("""
#   <style>
#     .custom-nav {
#       background: #e8f5e9; padding: 15px 0; border-radius: 10px;
#       display: flex; justify-content: center; gap: 60px; margin-bottom: 30px;
#       font-size: 18px; font-weight: 600;
#     }
#     .custom-nav a { text-decoration: none; color: #4C9D70; }
#     .custom-nav a:hover { color: #388e3c; text-decoration: underline; }
#   </style>
#   <div class="custom-nav">
#     <a href='/Home'>Home</a>
#     <a href='/Risk_Assessment'>Risk Assessment</a>
#     <a href='/Results'>Results</a>
#     <a href='/Recommendations'>Recommendations</a>
#   </div>
# """, unsafe_allow_html=True)

# # ── Load pipeline (polynomial step defined above) ──────────────────────────────
# @st.cache_resource
# def load_pipeline():
#     base = os.path.dirname(os.path.abspath(__file__))
#     return joblib.load(os.path.join(base, "best_gb_pipeline.pkl"))

# pipeline = load_pipeline()

# # ── Input Sections ─────────────────────────────────────────────────────────────
# with st.expander("👤 Personal Information", expanded=True):
#     age = st.number_input(
#         "Age", min_value=18, max_value=100,
#         value=18, step=1, format="%d", key="age"
#     )
#     gender = st.selectbox(
#         "Gender",
#         ["Select option", "Male", "Female"],
#         index=0, key="gender"
#     )
#     ever_married = st.selectbox(
#         "Ever Married?",
#         ["Select option", "Yes", "No"],
#         index=0, key="ever_married"
#     )
#     work_type = st.selectbox(
#         "Work Type",
#         ["Select option", "Private", "Self-employed", "Govt_job", "Never_worked"],
#         index=0, key="work_type"
#     )

# with st.expander("🩺 Health Information", expanded=True):
#     hypertension = st.radio(
#         "Do you have hypertension?",
#         ["Select option", "Yes", "No"],
#         index=0, key="hypertension"
#     )
#     heart_disease = st.radio(
#         "Do you have heart disease?",
#         ["Select option", "Yes", "No"],
#         index=0, key="heart_disease"
#     )
#     avg_glucose_level = st.number_input(
#         "Average Glucose Level (mg/dL)",
#         min_value=55.0, value=55.0, step=0.1,
#         key="avg_glucose_level"
#     )
#     smoking_status = st.selectbox(
#         "Smoking Status",
#         ["Select option", "never smoked", "formerly smoked", "smokes"],
#         index=0, key="smoking_status"
#     )

# # ── Consent & Disclaimer ───────────────────────────────────────────────────────
# st.markdown("### 📄 Consent and Disclaimer")
# st.write(
#     "This tool provides an estimate of stroke risk based on the information you provide. "
#     "It is not a diagnostic tool and should not replace professional medical advice. "
#     "By submitting, you agree to allow us to estimate your stroke risk."
# )
# st.checkbox("I agree to the terms and allow risk estimation", key="consent")

# # ── Calculate & Redirect ────────────────────────────────────────────────────────
# if st.button("Calculate Stroke Risk 📈"):
#     if not st.session_state.consent:
#         st.error("You must agree to the terms before proceeding!")
#     elif (
#         gender == "Select option"
#         or ever_married == "Select option"
#         or work_type == "Select option"
#         or hypertension == "Select option"
#         or heart_disease == "Select option"
#         or smoking_status == "Select option"
#         or age < 18
#         or avg_glucose_level <= 0
#     ):
#         st.error("Please complete all fields with valid values before submitting.")
#     else:
#         # Build raw feature array in same order as training
#         raw = [
#             age,
#             avg_glucose_level,
#             {"Yes":1,"No":0}[heart_disease],
#             {"Yes":1,"No":0}[hypertension],
#             {"Yes":1,"No":0}[ever_married],
#             {"never smoked":0,"formerly smoked":1,"smokes":2}[smoking_status],
#             {"Private":0,"Self-employed":1,"Govt_job":2,"Never_worked":4}[work_type],
#             {"Male":0,"Female":1}[gender]
#         ]
#         features = np.array(raw).reshape(1, -1)

#         prob = pipeline.predict_proba(features)[0, 1]

#         st.session_state.user_data = {
#             "age": age,
#             "avg_glucose_level": avg_glucose_level,
#             "heart_disease": heart_disease,
#             "hypertension": hypertension,
#             "ever_married": ever_married,
#             "smoking_status": smoking_status,
#             "work_type": work_type,
#             "gender": gender
#         }
#         st.session_state.prediction_prob = prob

#         st.switch_page("pages/Results.py")

# # ── Footer ────────────────────────────────────────────────────────────────────
# st.markdown("""
#   <style>
#     .custom-footer {
#       background-color: rgba(76,157,112,0.6); color: white;
#       padding: 30px 0; border-radius: 12px; margin-top: 40px;
#       text-align: center; font-size: 14px; width: 100%;
#     }
#     .custom-footer a { color: white; text-decoration: none; margin: 0 15px; }
#     .custom-footer a:hover { text-decoration: underline; }
#   </style>
#   <div class="custom-footer">
#       <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
#       <p>
#         <a href='/Home'>Home</a>
#         <a href='/Risk_Assessment'>Risk Assessment</a>
#         <a href='/Results'>Results</a>
#         <a href='/Recommendations'>Recommendations</a>
#       </p>
#       <p style="font-size:12px;">Developed by Victoria Mends</p>
#   </div>
# """, unsafe_allow_html=True)







# import streamlit as st
# import joblib
# import os
# import numpy as np

# # ── Page config & hide defaults ────────────────────────────────────────────────
# st.set_page_config(page_title="Stroke Risk Assessment", layout="wide")
# st.markdown("""
#     <style>
#       #MainMenu, footer, header {visibility: hidden;}
#       [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
#     </style>
# """, unsafe_allow_html=True)

# # ── Title & Navbar ─────────────────────────────────────────────────────────────
# st.title("📝 Stroke Risk Assessment")
# st.markdown("""
#   <style>
#     .custom-nav {
#       background: #e8f5e9; padding: 15px 0; border-radius: 10px;
#       display: flex; justify-content: center; gap: 60px; margin-bottom: 30px;
#       font-size: 18px; font-weight: 600;
#     }
#     .custom-nav a { text-decoration: none; color: #4C9D70; }
#     .custom-nav a:hover { color: #388e3c; text-decoration: underline; }
#   </style>
#   <div class="custom-nav">
#     <a href='/Home'>Home</a>
#     <a href='/Risk_Assessment'>Risk Assessment</a>
#     <a href='/Results'>Results</a>
#     <a href='/Recommendations'>Recommendations</a>
#   </div>
# """, unsafe_allow_html=True)

# # ── Load model ─────────────────────────────────────────────────────────────────
# @st.cache_resource
# def load_model():
#     base = os.path.dirname(os.path.abspath(__file__))
#     return joblib.load(os.path.join(base, "best_gb_model.pkl"))

# model = load_model()

# # ── Input Sections ─────────────────────────────────────────────────────────────
# with st.expander("👤 Personal Information", expanded=True):
#     age = st.number_input(
#         "Age", min_value=18, max_value=100,
#         value=18, step=1, format="%d", key="age"
#     )
#     gender = st.selectbox(
#         "Gender",
#         ["Select option", "Male", "Female"],
#         index=0, key="gender"
#     )
#     ever_married = st.selectbox(
#         "Ever Married?",
#         ["Select option", "Yes", "No"],
#         index=0, key="ever_married"
#     )
#     work_type = st.selectbox(
#         "Work Type",
#         ["Select option", "Private", "Self-employed", "Govt_job", "Never_worked"],
#         index=0, key="work_type"
#     )

# with st.expander("🩺 Health Information", expanded=True):
#     hypertension = st.radio(
#         "Do you have hypertension?",
#         ["Select option", "Yes", "No"],
#         index=0, key="hypertension"
#     )
#     heart_disease = st.radio(
#         "Do you have heart disease?",
#         ["Select option", "Yes", "No"],
#         index=0, key="heart_disease"
#     )
#     avg_glucose_level = st.number_input(
#         "Average Glucose Level (mg/dL)",
#         min_value=55.0, value=55.0, step=0.1,
#         key="avg_glucose_level"
#     )
#     smoking_status = st.selectbox(
#         "Smoking Status",
#         ["Select option", "never smoked", "formerly smoked", "smokes"],
#         index=0, key="smoking_status"
#     )

# # ── Consent & Disclaimer ───────────────────────────────────────────────────────
# st.markdown("### 📄 Consent and Disclaimer")
# st.write(
#     "This tool provides an estimate of stroke risk based on the information you provide. "
#     "It is not a diagnostic tool and should not replace professional medical advice. "
#     "By submitting, you agree to allow us to estimate your stroke risk."
# )
# st.checkbox("I agree to the terms and allow risk estimation", key="consent")

# # ── Calculate & Redirect ────────────────────────────────────────────────────────
# if st.button("Calculate Stroke Risk 📈"):
#     # Validation
#     if not st.session_state.consent:
#         st.error("You must agree to the terms before proceeding!")
#     elif (
#         gender == "Select option"
#         or ever_married == "Select option"
#         or work_type == "Select option"
#         or hypertension == "Select option"
#         or heart_disease == "Select option"
#         or smoking_status == "Select option"
#         or age < 18
#         or avg_glucose_level <= 0
#     ):
#         st.error("Please complete all fields with valid values before submitting.")
#     else:
#         # compute polynomial features
#         age_val = age
#         glu_val = avg_glucose_level
#         age_sq = age_val ** 2
#         glu_sq = glu_val ** 2
#         interaction = age_val * glu_val

#         # encoding maps
#         gender_map = {"Male": 1, "Female": 0}
#         married_map = {"Yes": 1, "No": 0}
#         work_map = {"Private": 2, "Self-employed": 3, "Govt_job": 0, "Never_worked": 1}
#         htn_map = {"Yes": 1, "No": 0}
#         heart_map = {"Yes": 1, "No": 0}
#         smoke_map = {"never smoked": 1, "formerly smoked": 0, "smokes": 2}

#         # build feature vector in the same order as training
#         features = np.array([[
#             heart_map[heart_disease],
#             htn_map[hypertension],
#             married_map[ever_married],
#             smoke_map[smoking_status],
#             work_map[work_type],
#             gender_map[gender],
#             age_val,
#             glu_val,
#             age_sq,
#             interaction,
#             glu_sq
#         ]], dtype=float)

#         # predict probability of stroke
#         prob = model.predict_proba(features)[0, 1]

#         # save for Results page
#         st.session_state.user_data = {
#             "age": age_val,
#             "gender": gender,
#             "ever_married": ever_married,
#             "work_type": work_type,
#             "hypertension": hypertension,
#             "heart_disease": heart_disease,
#             "avg_glucose_level": glu_val,
#             "smoking_status": smoking_status,
#         }
#         st.session_state.prediction_prob = prob

#         # navigate to your Results page
#         st.switch_page("pages/Results.py")

# # ── Footer ────────────────────────────────────────────────────────────────────
# st.markdown("""
#   <style>
#     .custom-footer {
#       background-color: rgba(76,157,112,0.6); color: white;
#       padding: 30px 0; border-radius: 12px; margin-top: 40px;
#       text-align: center; font-size: 14px; width: 100%;
#     }
#     .custom-footer a { color: white; text-decoration: none; margin: 0 15px; }
#     .custom-footer a:hover { text-decoration: underline; }
#   </style>
#   <div class="custom-footer">
#       <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
#       <p>
#         <a href='/Home'>Home</a>
#         <a href='/Risk_Assessment'>Risk Assessment</a>
#         <a href='/Results'>Results</a>
#         <a href='/Recommendations'>Recommendations</a>
#       </p>
#       <p style="font-size:12px;">Developed by Victoria Mends</p>
#   </div>
# """, unsafe_allow_html=True)








# import streamlit as st
# import pandas as pd
# import joblib, os          # ← makes engineer_feats importable

# # ----- place these THREE lines at the *very top* of the page -------------
# import feutils, __main__
# if not hasattr(__main__, "engineer_feats"):
#     setattr(__main__, "engineer_feats", feutils.engineer_feats)
# # -------------------------------------------------------------------------


# # ───────────────────────── Page config & CSS ─────────────────────────
# st.set_page_config(page_title="Stroke Risk Assessment", layout="wide")
# st.markdown("""
#     <style>
#       #MainMenu, footer, header {visibility: hidden;}
#       [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
#     </style>
# """, unsafe_allow_html=True)

# # ───────────────────────── Title & Navbar ────────────────────────────
# st.title("📝 Stroke Risk Assessment")
# st.markdown("""
#   <style>
#     .custom-nav{background:#e8f5e9;padding:15px 0;border-radius:10px;
#                 display:flex;justify-content:center;gap:60px;margin-bottom:30px;
#                 font-size:18px;font-weight:600;}
#     .custom-nav a{text-decoration:none;color:#4C9D70;}
#     .custom-nav a:hover{color:#388e3c;text-decoration:underline;}
#   </style>
#   <div class="custom-nav">
#     <a href='/Home'>Home</a><a href='/Risk_Assessment'>Risk Assessment</a>
#     <a href='/Results'>Results</a><a href='/Recommendations'>Recommendations</a>
#   </div>
# """, unsafe_allow_html=True)

# # ──────────────────────── Load pipeline ──────────────────────────────
# @st.cache_resource
# def load_pipeline():
#     import feutils, __main__
#     setattr(__main__, "engineer_feats", feutils.engineer_feats)

#     path = os.path.join(os.path.dirname(__file__), "stroke_stacking_pipeline.pkl")
#     return joblib.load(path)


# pipeline = load_pipeline()

# # ──────────────────────── Input Sections ─────────────────────────────
# with st.expander("👤 Personal Information", expanded=True):
#     age          = st.number_input("Age", 18, 100, 45, step=1)
#     gender       = st.selectbox("Gender", ["Select option","Male","Female"])
#     ever_married = st.selectbox("Ever Married?", ["Select option","Yes","No"])
#     work_type    = st.selectbox("Work Type",
#                                 ["Select option","Private","Self-employed",
#                                  "Govt_job","Never_worked"])

# with st.expander("🩺 Health Information", expanded=True):
#     hypertension   = st.radio("Hypertension",  ["Select option","Yes","No"], horizontal=True)
#     heart_disease  = st.radio("Heart Disease", ["Select option","Yes","No"], horizontal=True)
#     avg_glucose    = st.number_input("Average Glucose Level (mg/dL)", 55.0, 400.0, 110.0)
#     smoking_status = st.selectbox("Smoking Status",
#                                   ["Select option","never smoked","formerly smoked","smokes"])

# # ───────────────────── Consent & Disclaimer ──────────────────────────
# st.markdown("### 📄 Consent and Disclaimer")
# st.write(
#     "This tool provides an estimate of stroke risk based on the information you provide. "
#     "It is not a diagnostic tool and should not replace professional medical advice."
# )
# agreed = st.checkbox("I agree to the terms and allow risk estimation")

# # ─────────────────────── Calculate & Redirect ───────────────────────
# if st.button("Calculate Stroke Risk 📈"):
#     if not agreed:
#         st.error("You must agree to the terms before proceeding.")
#     elif any(x.startswith("Select") for x in
#              [gender, ever_married, work_type, hypertension, heart_disease, smoking_status]):
#         st.error("Please complete all fields.")
#     else:
#         user_data = {
#             "gender":            gender,
#             "age":               age,
#             "avg_glucose_level": avg_glucose,
#             "hypertension":      hypertension,
#             "heart_disease":     heart_disease,
#             "ever_married":      ever_married,
#             "work_type":         work_type,
#             "smoking_status":    smoking_status
#         }
#         prob = float(pipeline.predict_proba(pd.DataFrame([user_data]))[0, 1])

#         st.session_state.user_data       = user_data
#         st.session_state.prediction_prob = prob
#         st.switch_page("pages/Results.py")

# # ─────────────────────────── Footer ────────────────────────────────
# st.markdown("""
#   <style>
#     .custom-footer{background:rgba(76,157,112,0.6);color:white;padding:30px 0;
#                    border-radius:12px;margin-top:40px;text-align:center;font-size:14px;}
#     .custom-footer a{color:white;text-decoration:none;margin:0 15px;}
#     .custom-footer a:hover{text-decoration:underline;}
#   </style>
#   <div class='custom-footer'>
#       <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
#       <p>
#         <a href='/Home'>Home</a><a href='/Risk_Assessment'>Risk Assessment</a>
#         <a href='/Results'>Results</a><a href='/Recommendations'>Recommendations</a>
#       </p>
#       <p style='font-size:12px;'>Developed by Victoria Mends</p>
#   </div>
# """, unsafe_allow_html=True)










# import streamlit as st
# import joblib
# import os
# import numpy as np

# # ── Page config & hide defaults ────────────────────────────────────────────────
# st.set_page_config(page_title="Stroke Risk Assessment", layout="wide")
# st.markdown("""
#     <style>
#       #MainMenu, footer, header {visibility: hidden;}
#       [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
#     </style>
# """, unsafe_allow_html=True)

# # ── Title & Navbar ─────────────────────────────────────────────────────────────
# st.title("📝 Stroke Risk Assessment")
# st.markdown("""
#   <style>
#     .custom-nav {
#       background: #e8f5e9; padding: 15px 0; border-radius: 10px;
#       display: flex; justify-content: center; gap: 60px; margin-bottom: 30px;
#       font-size: 18px; font-weight: 600;
#     }
#     .custom-nav a { text-decoration: none; color: #4C9D70; }
#     .custom-nav a:hover { color: #388e3c; text-decoration: underline; }
#   </style>
#   <div class="custom-nav">
#     <a href='/Home'>Home</a>
#     <a href='/Risk_Assessment'>Risk Assessment</a>
#     <a href='/Results'>Results</a>
#     <a href='/Recommendations'>Recommendations</a>
#   </div>
# """, unsafe_allow_html=True)

# # ── Load model ─────────────────────────────────────────────────────────────────
# @st.cache_resource
# # def load_model():
# #     base = os.path.dirname(os.path.abspath(__file__))
# #     return joblib.load(os.path.join(base, "best_gb_model.pkl"))

# # model = load_model()

# def load_artifacts():
#     base = os.path.dirname(os.path.abspath(__file__))
#     scaler = joblib.load(os.path.join(base, "scaler.pkl"))
#     model  = joblib.load(os.path.join(base, "best_gb_model.pkl"))
#     return scaler, model

# scaler, model = load_artifacts()

# # ── Input Sections ─────────────────────────────────────────────────────────────
# with st.expander("👤 Personal Information", expanded=True):
#     age = st.number_input("Age", min_value=18, max_value=100,
#                           value=18, step=1, format="%d", key="age")
#     gender = st.selectbox("Gender",
#                           ["Select option", "Male", "Female"],
#                           index=0, key="gender")
#     ever_married = st.selectbox("Ever Married?",
#                                 ["Select option", "Yes", "No"],
#                                 index=0, key="ever_married")
#     work_type = st.selectbox("Work Type",
#                              ["Select option",
#                               "Private", "Self-employed",
#                               "Govt_job", "Never_worked"],
#                              index=0, key="work_type")

# with st.expander("🩺 Health Information", expanded=True):
#     hypertension = st.radio("Do you have hypertension?",
#                             ["Select option", "Yes", "No"],
#                             index=0, key="hypertension")
#     heart_disease = st.radio("Do you have heart disease?",
#                              ["Select option", "Yes", "No"],
#                              index=0, key="heart_disease")
#     avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)",
#                                         min_value=55.0, value=55.0,
#                                         step=0.1, key="avg_glucose_level")
#     smoking_status = st.selectbox("Smoking Status",
#                                   ["Select option",
#                                    "never smoked",
#                                    "formerly smoked",
#                                    "smokes"],
#                                   index=0, key="smoking_status")

# # ── Consent & Disclaimer ───────────────────────────────────────────────────────
# st.markdown("### 📄 Consent and Disclaimer")
# st.write(
#     "This tool provides an estimate of stroke risk based on the information you provide. "
#     "It is not a diagnostic tool and should not replace professional medical advice. "
#     "By submitting, you agree to allow us to estimate your stroke risk."
# )
# st.checkbox("I agree to the terms and allow risk estimation", key="consent")

# # ── Calculate & Redirect ────────────────────────────────────────────────────────
# if st.button("Calculate Stroke Risk 📈"):
#     # validation
#     if not st.session_state.consent:
#         st.error("You must agree to the terms before proceeding!")
#     elif (
#         gender == "Select option"
#         or ever_married == "Select option"
#         or work_type == "Select option"
#         or hypertension == "Select option"
#         or heart_disease == "Select option"
#         or smoking_status == "Select option"
#         or age < 18
#         or avg_glucose_level <= 0
#     ):
#         st.error("Please complete all fields with valid values before submitting.")
#     else:
#         # # Compute polynomial terms
#         # age_sq       = age ** 2
#         # glucose_sq   = avg_glucose_level ** 2
#         # interaction  = age * avg_glucose_level

#         # # Encoding maps (exactly as in training)
#         # gender_map  = {"Male": 0, "Female": 1}
#         # married_map = {"Yes": 1, "No": 0}
#         # work_map    = {
#         #     "Private": 0,
#         #     "Self-employed": 1,
#         #     "Govt_job": 2,
#         #     "Never_worked": 3
#         # }
#         # htn_map     = {"Yes": 1, "No": 0}
#         # heart_map   = {"Yes": 1, "No": 0}
#         # smoke_map   = {
#         #     "never smoked": 0,
#         #     "formerly smoked": 1,
#         #     "smokes": 2
#         # }

#         # # Build feature vector in the same order you trained on
#         # features = np.array([[
#         #     heart_map[heart_disease],                # heart_disease
#         #     htn_map[hypertension],                   # hypertension
#         #     married_map[ever_married],               # married_map[ever_married]
#         #     smoke_map[smoking_status],               # smoking_map[smoking_status]
#         #     work_map[work_type],                     # work_type_map[work_type]
#         #     gender_map[gender],                      # gender_map[gender]
#         #     age,                                     # age
#         #     avg_glucose_level,                       # avg_glucose_level
#         #     age_sq,                                  # age_squared
#         #     interaction,                             # interaction
#         #     glucose_sq                               # glucose_squared
#         # ]], dtype=float)

#         # # Predict probability of stroke
#         # prob = model.predict_proba(features)[0, 1]

#           # Compute polynomial terms
#         age_sq      = age ** 2
#         glucose_sq  = avg_glucose_level ** 2
#         interaction = age * avg_glucose_level

#         # Encoding maps (exactly as in training)
#         gender_map  = {"Male": 0, "Female": 1}
#         married_map = {"Yes": 1, "No": 0}
#         work_map    = {
#             "Private": 0,
#             "Self-employed": 1,
#             "Govt_job": 2,
#             "Never_worked": 3
#         }
#         htn_map     = {"Yes": 1, "No": 0}
#         heart_map   = {"Yes": 1, "No": 0}
#         smoke_map   = {
#             "never smoked": 0,
#             "formerly smoked": 1,
#             "smokes": 2
#         }

#         # Build feature vector in training order
#         features = np.array([[  
#             heart_map[heart_disease],  # heart_disease
#             htn_map[hypertension],     # hypertension
#             married_map[ever_married],  # ever_married
#             smoke_map[smoking_status],  # smoking_status
#             work_map[work_type],        # work_type
#             gender_map[gender],         # gender
#             age,                        # age
#             avg_glucose_level,          # avg_glucose_level
#             age_sq,                     # age_squared
#             interaction,                # age_glucose interaction
#             glucose_sq                  # glucose_squared
#         ]], dtype=float)

#         # Scale features before prediction
#         scaled_feats = scaler.transform(features)

#         # Predict probability of stroke using GB model
#         prob = model.predict_proba(scaled_feats)[0, 1]

#         # Save for Results.py
#         st.session_state.user_data = {
#             "age": age,
#             "gender": gender,
#             "ever_married": ever_married,
#             "work_type": work_type,
#             "hypertension": hypertension,
#             "heart_disease": heart_disease,
#             "avg_glucose_level": avg_glucose_level,
#             "smoking_status": smoking_status
#         }
#         st.session_state.prediction_prob = prob

#         # Navigate to Results page
#         st.switch_page("pages/Results.py")

# # ── Footer ────────────────────────────────────────────────────────────────────
# st.markdown("""
#   <style>
#     .custom-footer {
#       background-color: rgba(76,157,112,0.6); color: white;
#       padding: 30px 0; border-radius: 12px; margin-top: 40px;
#       text-align: center; font-size: 14px; width: 100%;
#     }
#     .custom-footer a { color: white; text-decoration: none; margin: 0 15px; }
#     .custom-footer a:hover { text-decoration: underline; }
#   </style>
#   <div class="custom-footer">
#       <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
#       <p>
#         <a href='/Home'>Home</a>
#         <a href='/Risk_Assessment'>Risk Assessment</a>
#         <a href='/Results'>Results</a>
#         <a href='/Recommendations'>Recommendations</a>
#       </p>
#       <p style="font-size:12px;">Developed by Victoria Mends</p>
#   </div>
# """, unsafe_allow_html=True)







# import streamlit as st
# import joblib
# import os
# import numpy as np

# # ── Page config & hide defaults ────────────────────────────────────────────────
# st.set_page_config(page_title="Stroke Risk Assessment", layout="wide")
# st.markdown("""
#     <style>
#       #MainMenu, footer, header {visibility: hidden;}
#       [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
#     </style>
# """, unsafe_allow_html=True)

# # ── Title & Navbar ─────────────────────────────────────────────────────────────
# st.title("📝 Stroke Risk Assessment")
# st.markdown("""
#   <style>
#     .custom-nav {
#       background: #e8f5e9; padding: 15px 0; border-radius: 10px;
#       display: flex; justify-content: center; gap: 60px; margin-bottom: 30px;
#       font-size: 18px; font-weight: 600;
#     }
#     .custom-nav a { text-decoration: none; color: #4C9D70; }
#     .custom-nav a:hover { color: #388e3c; text-decoration: underline; }
#   </style>
#   <div class="custom-nav">
#     <a href='/Home'>Home</a>
#     <a href='/Risk_Assessment'>Risk Assessment</a>
#     <a href='/Results'>Results</a>
#     <a href='/Recommendations'>Recommendations</a>
#   </div>
# """, unsafe_allow_html=True)

# # ── Load model ─────────────────────────────────────────────────────────────────
# @st.cache_resource
# # def load_model():
# #     base = os.path.dirname(os.path.abspath(__file__))
# #     return joblib.load(os.path.join(base, "best_gb_model.pkl"))

# # model = load_model()

# def load_artifacts():
#     base = os.path.dirname(os.path.abspath(__file__))
#     scaler = joblib.load(os.path.join(base, "scaler.pkl"))
#     model  = joblib.load(os.path.join(base, "best_gb_model.pkl"))
#     return scaler, model

# scaler, model = load_artifacts()

# # ── Input Sections ─────────────────────────────────────────────────────────────
# with st.expander("👤 Personal Information", expanded=True):
#     age = st.number_input(
#         "Age", min_value=18, max_value=100,
#         value=18, step=1, format="%d", key="age"
#     )
#     gender = st.selectbox(
#         "Gender",
#         ["Select option", "Male", "Female"],
#         index=0, key="gender"
#     )
#     ever_married = st.selectbox(
#         "Ever Married?",
#         ["Select option", "Yes", "No"],
#         index=0, key="ever_married"
#     )
#     work_type = st.selectbox(
#         "Work Type",
#         ["Select option", "Private", "Self-employed", "Govt_job", "Never_worked"],
#         index=0, key="work_type"
#     )

# with st.expander("🩺 Health Information", expanded=True):
#     hypertension = st.radio(
#         "Do you have hypertension?",
#         ["Select option", "Yes", "No"],
#         index=0, key="hypertension"
#     )
#     heart_disease = st.radio(
#         "Do you have heart disease?",
#         ["Select option", "Yes", "No"],
#         index=0, key="heart_disease"
#     )
#     avg_glucose_level = st.number_input(
#         "Average Glucose Level (mg/dL)",
#         min_value=55.0, value=55.0, step=0.1,
#         key="avg_glucose_level"
#     )
#     smoking_status = st.selectbox(
#         "Smoking Status",
#         ["Select option", "never smoked", "formerly smoked", "smokes"],
#         index=0, key="smoking_status"
#     )

# # ── Consent & Disclaimer ───────────────────────────────────────────────────────
# st.markdown("### 📄 Consent and Disclaimer")
# st.write(
#     "This tool provides an estimate of stroke risk based on the information you provide. "
#     "It is not a diagnostic tool and should not replace professional medical advice. "
#     "By submitting, you agree to allow us to estimate your stroke risk."
# )
# st.checkbox("I agree to the terms and allow risk estimation", key="consent")

# # ── Calculate & Redirect ────────────────────────────────────────────────────────
# if st.button("Calculate Stroke Risk 📈"):
#     # basic validation
#     if not st.session_state.consent:
#         st.error("You must agree to the terms before proceeding!")
#     elif (
#         st.session_state.gender == "Select option"
#         or st.session_state.ever_married == "Select option"
#         or st.session_state.work_type == "Select option"
#         or st.session_state.hypertension == "Select option"
#         or st.session_state.heart_disease == "Select option"
#         or st.session_state.smoking_status == "Select option"
#         or st.session_state.age < 18
#         or st.session_state.avg_glucose_level <= 0
#     ):
#         st.error("Please complete all fields with valid values before submitting.")
#     else:
#         # compute polynomial features
#         age_val   = st.session_state.age
#         glu_val   = st.session_state.avg_glucose_level
#         age_sq    = age_val ** 2
#         glu_sq    = glu_val ** 2
#         interaction = age_val * glu_val

#         # encoding maps
#         gender_map      = {"Male": 1, "Female": 0}
#         married_map     = {"Yes": 1, "No": 0}
#         work_map        = {"Private": 2, "Self-employed": 3, "Govt_job": 0, "Never_worked": 1}
#         htn_map         = {"Yes": 1, "No": 0}
#         heart_map       = {"Yes": 1, "No": 0}
#         smoke_map       = {"never smoked": 1, "formerly smoked": 0, "smokes": 2}

#         # build feature vector
#         features = np.array([
#             heart_map[st.session_state.heart_disease],
#             htn_map[st.session_state.hypertension],
#             married_map[st.session_state.ever_married],
#             smoke_map[st.session_state.smoking_status],
#             work_map[st.session_state.work_type],
#             gender_map[st.session_state.gender],
#             age_val,
#             glu_val,
#             age_sq,
#             interaction,
#             glu_sq
#         ], dtype=float).reshape(1, -1)

#         # predict probability
#         prob = model.predict_proba(features)[0][1]
# +       # scale features exactly as during training
# +       features = scaler.transform(features)
# +       # predict probability on the scaled data
# +        
#         prob = model.predict_proba(features)[0][1]

#         # save for Results.py
#         st.session_state.user_data       = {
#             "age": age_val,
#             "gender": st.session_state.gender,
#             "ever_married": st.session_state.ever_married,
#             "work_type": st.session_state.work_type,
#             "hypertension": st.session_state.hypertension,
#             "heart_disease": st.session_state.heart_disease,
#             "avg_glucose_level": glu_val,
#             "smoking_status": st.session_state.smoking_status
#         }
#         st.session_state.prediction_prob = prob

#         # navigate to your Results page
#         st.switch_page("pages/Results.py")

# # ── Footer ────────────────────────────────────────────────────────────────────
# st.markdown("""
#   <style>
#     .custom-footer {
#       background-color: rgba(76,157,112,0.6); color: white;
#       padding: 30px 0; border-radius: 12px; margin-top: 40px;
#       text-align: center; font-size: 14px; width: 100%;
#     }
#     .custom-footer a { color: white; text-decoration: none; margin: 0 15px; }
#     .custom-footer a:hover { text-decoration: underline; }
#   </style>
#   <div class="custom-footer">
#       <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
#       <p>
#         <a href='/Home'>Home</a>
#         <a href='/Risk_Assessment'>Risk Assessment</a>
#         <a href='/Results'>Results</a>
#         <a href='/Recommendations'>Recommendations</a>
#       </p>
#       <p style="font-size:12px;">Developed by Victoria Mends</p>
#   </div>
# """, unsafe_allow_html=True)
