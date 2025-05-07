import os
import joblib
import streamlit as st
import numpy as np
import shap
import plotly.graph_objects as go

# ── Page config must be first ─────────────────────────────────────────────────



# ── Polynomial feature helper ─────────────────────────────────────────────────
def add_poly(X):
    # X shape: (n_samples, 8) raw features
    age    = X[:, 0]
    glu    = X[:, 1]
    age_sq = age ** 2
    inter  = age * glu
    glu_sq = glu ** 2
    return np.c_[X, age_sq, inter, glu_sq]

# ── Scaler parameters from training ────────────────────────────────────────────
SCALER_MEAN = np.array([
    47.4572, 106.1478,  # age, glucose
    0.0482, 0.0513,     # heart_disease, hypertension
    0.5527, 0.5431,     # ever_married, smoking_status
    2.1356, 0.5064,     # work_type, gender
    1850.37, 5067.84, 11645.2  # age_sq, inter, glu_sq
])
SCALER_SCALE = np.array([
    15.6753, 26.8145,
    0.2141, 0.2206,
    0.4974, 0.4983,
    0.9082, 0.4999,
    2978.41, 6144.78, 10795.6
])

# ── Load bare model for prediction and SHAP ───────────────────────────────────
@st.cache_resource
# Cache the model loading; no hashing of large objects
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "best_gb_model.pkl")
    return joblib.load(path)

model = load_model()

# ── SHAP explainer ─────────────────────────────────────────────────────────────
@st.cache_resource
# Use underscore to prevent hashing the model object
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(model)

# ── Page config & CSS ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Stroke Risk Results", layout="wide")
st.markdown("""
    <style>
      #MainMenu, footer, header {visibility: hidden;}
      [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
      .header-container {
        background: #4C9D70; padding: 15px 0; text-align: center;
        border-radius: 8px; margin-bottom: 20px;
      }
      .header-container h1 { color: white; margin: 0; }
      .custom-nav {
        background: #E8F5E9; padding: 10px; border-radius: 8px;
        display: flex; justify-content: center; gap: 40px;
        margin-bottom: 30px; font-size: 16px; font-weight: 600;
      }
      .custom-nav a { text-decoration: none; color: #4C9D70; }
      .custom-nav a:hover { color: #388E3C; text-decoration: underline; }
      @media (prefers-color-scheme: dark) {
        .header-container { background: #1f2c2f !important; }
        .custom-nav { background: #2c2c2e !important; }
        .custom-nav a { color: #ddd !important; }
        .custom-nav a:hover { color: #fff !important; }
      }
    </style>
""", unsafe_allow_html=True)

# ── Header & Navbar ───────────────────────────────────────────────────────────
st.markdown("""
  <div class="header-container">
    <h1>📊 Stroke Risk Results</h1>
  </div>
  <div class="custom-nav">
    <a href='/Home'>Home</a>
    <a href='/Risk_Assessment'>Risk Assessment</a>
    <a href='/Results'>Results</a>
    <a href='/Recommendations'>Recommendations</a>
  </div>
""", unsafe_allow_html=True)

# ── Display results & SHAP contributions ───────────────────────────────────────
if "user_data" in st.session_state and "prediction_prob" in st.session_state:
    prob = st.session_state.prediction_prob
    pct  = prob * 100

    st.markdown(f"### 🧠 Your Stroke Percentage Risk: **{pct:.2f}%**")
    st.write("---")

    UD = st.session_state.user_data
    # Rebuild raw feature vector in training order
    raw = [
    UD["age"],
    UD["avg_glucose_level"],
    1 if UD["heart_disease"] == "Yes" else 0,
    1 if UD["hypertension"] == "Yes" else 0,
    1 if UD["ever_married"] == "Yes" else 0,
    {"formerly smoked": 0, "smokes": 2, "never smoked": 1}[UD["smoking_status"]],
    {"Private": 2, "Govt_job": 0, "Self-employed": 3, "Never_worked": 1}[UD["work_type"]],
    {"Male": 1, "Female": 0}[UD["gender"]]
]
    X_raw    = np.array(raw).reshape(1, -1)
    X_poly   = add_poly(X_raw)
    X_scaled = (X_poly - SCALER_MEAN) / SCALER_SCALE

    # SHAP values
    sv        = explainer.shap_values(X_scaled)
    shap_vals = sv[1][0] if isinstance(sv, list) else sv[0]
    vals      = np.abs(shap_vals[:8])
    contrib   = vals / vals.sum() * prob

    feature_names = [
      "Age", "Avg Glucose", "Heart Disease", "Hypertension",
      "Ever Married", "Smoking Status", "Work Type", "Gender"
    ]
    palette = ["brown","gold","steelblue","purple"]
    colors  = [palette[i % len(palette)] for i in range(len(feature_names))]

    # Contribution bar chart
    bar_fig = go.Figure(
        go.Bar(x=feature_names,
               y=contrib * 100,
               marker=dict(color=colors),
               text=[f"{v*100:.2f}%" for v in contrib],
               textposition="outside")
    )
    bar_fig.update_layout(
        template="plotly_white",
        title="How Each Input Contributed to Your Total Risk",
        yaxis=dict(title="Contribution to Risk (%)", range=[0,100], ticksuffix="%"),
        xaxis=dict(tickangle=-45),
        margin=dict(t=60, b=120)
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    # Gauge chart
    r = int(255 * prob)
    g = int(255 * (1 - prob))
    bar_color = f"rgb({r},{g},0)"
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={'suffix': "%"},
            title={'text': "Overall Stroke Risk (%)"},
            gauge={
                'axis': {'range': [0,100], 'ticksuffix': '%'},
                'bar': {'color': bar_color},
                'steps': [{'range': [0,50], 'color': 'green'}, {'range': [50,100], 'color': 'red'}]
            }
        )
    )
    gauge_fig.update_layout(template="plotly_white", margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔙 Back to Assessment"):
            st.switch_page("pages/Risk_Assessment.py")
    with col2:
        if st.button("📘 Recommendations"):
            st.switch_page("pages/Recommendations.py")
else:
    st.warning("No input data found. Please complete the Risk Assessment first.")

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







# # pages/Results.py

# import os
# import joblib
# import streamlit as st
# import numpy as np
# import shap
# import plotly.graph_objects as go

# # ── Page config & hide defaults ────────────────────────────────────────────────
# st.set_page_config(page_title="Stroke Risk Results", layout="wide")
# st.markdown("""
#     <style>
#       #MainMenu, footer, header {visibility: hidden;}
#       [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}

#       .header-container { background: #4C9D70; padding: 15px 0; text-align: center;
#                           border-radius: 8px; margin-bottom: 20px; }
#       .header-container h1 { color: white; margin: 0; }

#       .custom-nav { background: #E8F5E9; padding: 10px; border-radius: 8px;
#                     display: flex; justify-content: center; gap: 40px;
#                     margin-bottom: 30px; font-size: 16px; font-weight: 600; }
#       .custom-nav a { text-decoration: none; color: #4C9D70; }
#       .custom-nav a:hover { color: #388E3C; text-decoration: underline; }

#       @media (prefers-color-scheme: dark) {
#         .header-container { background: #1f2c2f !important; }
#         .custom-nav { background: #2c2c2e !important; }
#         .custom-nav a { color: #ddd !important; }
#         .custom-nav a:hover { color: #fff !important; }
#       }
#     </style>
# """, unsafe_allow_html=True)

# # ── Title & Navbar ─────────────────────────────────────────────────────────────
# st.markdown("""
#   <div class="header-container">
#     <h1>📊 Stroke Risk Results</h1>
#   </div>
#   <div class="custom-nav">
#     <a href='/Home'>Home</a>
#     <a href='/Risk_Assessment'>Risk Assessment</a>
#     <a href='/Results'>Results</a>
#     <a href='/Recommendations'>Recommendations</a>
#   </div>
# """, unsafe_allow_html=True)

# # ── Load full pipeline & explainer ─────────────────────────────────────────────
# @st.cache_resource
# def load_pipeline():
#     base = os.path.dirname(os.path.abspath(__file__))
#     return joblib.load(os.path.join(base, "best_gb_pipeline.pkl"))

# @st.cache_resource
# def load_explainer(pipeline):
#     model = pipeline.named_steps["gb"]
#     return shap.TreeExplainer(model)

# pipeline  = load_pipeline()
# explainer = load_explainer(pipeline)

# # ── Display results & SHAP contributions ───────────────────────────────────────
# if "user_data" in st.session_state and "prediction_prob" in st.session_state:
#     prob = st.session_state.prediction_prob
#     pct  = prob * 100

#     st.markdown(f"### 🧠 Your Stroke Percentage Risk: **{pct:.2f}%**")
#     st.write("---")

#     UD = st.session_state.user_data
#     # rebuild raw feature array
#     raw_list = [
#         UD["age"],
#         UD["avg_glucose_level"],
#         {"Yes":1,"No":0}[UD["heart_disease"]],
#         {"Yes":1,"No":0}[UD["hypertension"]],
#         {"Yes":1,"No":0}[UD["ever_married"]],
#         {"never smoked":0,"formerly smoked":1,"smokes":2}[UD["smoking_status"]],
#         {"Private":0,"Self-employed":1,"Govt_job":2,"Never_worked":4}[UD["work_type"]],
#         {"Male":0,"Female":1}[UD["gender"]]
#     ]
#     X_raw = np.array(raw_list).reshape(1, -1)

#     # apply polynomial & scaling (skip SMOTE at inference)
#     X_poly   = pipeline.named_steps["poly"].transform(X_raw)
#     X_scaled = pipeline.named_steps["scale"].transform(X_poly)

#     # SHAP values
#     sv = explainer.shap_values(X_scaled)
#     shap_vals = sv[1][0] if isinstance(sv, list) else sv[0]
#     vals     = np.abs(shap_vals[:8])
#     contrib  = vals / vals.sum() * prob

#     feature_names = [
#       "Age", "Avg Glucose", "Heart Disease", "Hypertension",
#       "Ever Married", "Smoking Status", "Work Type", "Gender"
#     ]
#     palette = ["brown","gold","steelblue","purple"]
#     colors  = [palette[i % len(palette)] for i in range(len(feature_names))]

#     # Bar chart
#     bar_fig = go.Figure(
#         go.Bar(
#             x=feature_names,
#             y=contrib * 100,
#             marker=dict(color=colors),
#             text=[f"{v*100:.2f}%" for v in contrib],
#             textposition="outside"
#         )
#     )
#     bar_fig.update_layout(
#         template="plotly_white",
#         title="How Each Input Contributed to Your Total Risk",
#         yaxis=dict(title="Contribution to Risk (%)", range=[0,100], ticksuffix="%"),
#         xaxis=dict(tickangle=-45),
#         margin=dict(t=60, b=120),
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)"
#     )
#     st.plotly_chart(bar_fig, use_container_width=True)

#     # Gauge chart
#     r = int(255 * prob)
#     g = int(255 * (1 - prob))
#     bar_color = f"rgb({r},{g},0)"

#     gauge_fig = go.Figure(
#         go.Indicator(
#             mode="gauge+number",
#             value=pct,
#             number={'suffix': "%"},
#             title={'text': "Overall Stroke Risk (%)"},
#             gauge={
#                 'axis': {'range': [0,100], 'ticksuffix': '%'},
#                 'bar': {'color': bar_color},
#                 'steps': [
#                     {'range': [0,50],  'color': 'green'},
#                     {'range': [50,100],'color': 'red'}
#                 ]
#             }
#         )
#     )
#     gauge_fig.update_layout(
#         template="plotly_white",
#         margin=dict(t=40, b=0, l=0, r=0),
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)"
#     )
#     st.plotly_chart(gauge_fig, use_container_width=True)

#     # Navigation buttons
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("🔙 Back to Assessment"):
#             st.switch_page("pages/Risk_Assessment.py")
#     with col2:
#         if st.button("📘 Recommendations"):
#             st.switch_page("pages/Recommendations.py")

# else:
#     st.warning("No input data found. Please complete the Risk Assessment first.")

# # ── Footer ────────────────────────────────────────────────────────────────────
# st.markdown("""
#   <style>
#     .custom-footer { background: rgba(76,157,112,0.6); color: white; padding: 30px 0;
#                      border-radius: 12px; margin-top: 40px; text-align: center; font-size: 14px; }
#     .custom-footer a { color: white; text-decoration: none; margin: 0 15px; }
#     .custom-footer a:hover { text-decoration: underline; }
#   </style>
#   <div class='custom-footer'>
#     <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
#     <p>
#       <a href='/Home'>Home</a>
#       <a href='/Risk_Assessment'>Risk Assessment</a>
#       <a href='/Results'>Results</a>
#       <a href='/Recommendations'>Recommendations</a>
#     </p>
#     <p style='font-size:12px; margin-top:10px;'>Developed by Victoria Mends</p>
#   </div>
# """, unsafe_allow_html=True)









# import streamlit as st
# import os
# import joblib
# import numpy as np
# import shap
# import plotly.graph_objects as go

# # ── Page config & hide defaults ────────────────────────────────────────────────
# st.set_page_config(page_title="Stroke Risk Results", layout="wide")
# st.markdown("""
#     <style>
#       /* Hide default Streamlit chrome */
#       #MainMenu, footer, header {visibility: hidden;}
#       [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}

#       /* Header bar */
#       .header-container {
#         background: #4C9D70;
#         padding: 15px 0;
#         text-align: center;
#         border-radius: 8px;
#         margin-bottom: 20px;
#       }
#       .header-container h1 {
#         color: white;
#         margin: 0;
#       }

#       /* Navigation bar */
#       .custom-nav {
#         background: #E8F5E9;
#         padding: 10px;
#         border-radius: 8px;
#         display: flex;
#         justify-content: center;
#         gap: 40px;
#         margin-bottom: 30px;
#         font-size: 16px;
#         font-weight: 600;
#       }
#       .custom-nav a {
#         text-decoration: none;
#         color: #4C9D70;
#       }
#       .custom-nav a:hover {
#         color: #388E3C;
#         text-decoration: underline;
#       }

#       /* Dark-mode overrides */
#       @media (prefers-color-scheme: dark) {
#         .header-container { background: #1f2c2f !important; }
#         .custom-nav { background: #2c2c2e !important; }
#         .custom-nav a { color: #ddd !important; }
#         .custom-nav a:hover { color: #fff !important; }
#       }
#     </style>
# """, unsafe_allow_html=True)

# # ── Title & Navbar ─────────────────────────────────────────────────────────────
# st.markdown("""
#   <div class="header-container">
#     <h1>📊 Stroke Risk Results</h1>
#   </div>
#   <div class="custom-nav">
#     <a href='/Home'>Home</a>
#     <a href='/Risk_Assessment'>Risk Assessment</a>
#     <a href='/Results'>Results</n    <a href='/Recommendations'>Recommendations</a>
#   </div>
# """, unsafe_allow_html=True)

# # ── Load trained model ─────────────────────────────────────────────────────────
# @st.cache_resource
# def load_model():
#     base = os.path.dirname(os.path.abspath(__file__))
#     path = os.path.join(base, "best_gb_model.pkl")
#     if not os.path.exists(path):
#         st.error(f"⚠️ Model not found at `{path}`")
#         st.stop()
#     return joblib.load(path)

# model = load_model()

# # ── Compute SHAP explainer once ────────────────────────────────────────────────
# @st.cache_resource
# def get_explainer(_model):
#     return shap.TreeExplainer(_model)

# explainer = get_explainer(model)

# # ── Display results & SHAP contributions ───────────────────────────────────────
# if "user_data" in st.session_state and "prediction_prob" in st.session_state:
#     prob = st.session_state.prediction_prob

#     st.markdown(f"### 🧠 Your Stroke Percentage Risk: **{prob*100:.2f}%**")
#     st.write("---")

#     # rebuild feature vector
#     UD = st.session_state.user_data
#     age, glu = UD["age"], UD["avg_glucose_level"]
#     age_sq, glu_sq = age**2, glu**2
#     interaction = age * glu

#     X = np.array([[
#       {"Yes":1,"No":0}[UD["heart_disease"]],
#       {"Yes":1,"No":0}[UD["hypertension"]],
#       {"Yes":1,"No":0}[UD["ever_married"]],
#       {"never smoked":1,"formerly smoked":0,"smokes":2}[UD["smoking_status"]],
#       {"Private":2,"Self-employed":3,"Govt_job":0,"Never_worked":1}[UD["work_type"]],
#       {"Male":1,"Female":0}[UD["gender"]],
#       age, glu, age_sq, interaction, glu_sq
#     ]])

#     sv = explainer.shap_values(X)
#     shap_vals = sv[1][0] if isinstance(sv, list) else sv[0]
#     vals = np.abs(shap_vals[:8])
#     contrib = vals / vals.sum() * prob

#     feature_names = [
#       "Heart Disease", "Hypertension", "Ever Married",
#       "Smoking Status", "Work Type", "Gender",
#       "Age", "Avg Glucose"
#     ]
#     palette = ["brown","gold","steelblue","purple"]
#     colors = [palette[i % len(palette)] for i in range(len(feature_names))]

#     # Bar chart with percentages scale
#     bar_fig = go.Figure(
#         go.Bar(
#             x=feature_names,
#             y=contrib * 100,
#             marker=dict(color=colors),
#             text=[f"{v*100:.2f}%" for v in contrib],
#             textposition="outside"
#         )
#     )
#     bar_fig.update_layout(
#         template="plotly_white",
#         title="How Each Input Contributed to Your Total Risk",
#         yaxis=dict(
#             title="Contribution to Risk (%)",
#             range=[0, 100],
#             ticksuffix="%"
#         ),
#         xaxis=dict(tickangle=-45),
#         margin=dict(t=60, b=120),
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)"
#     )
#     st.plotly_chart(bar_fig, use_container_width=True)

#     # Gauge chart with green-red zones
#     # define pointer color dynamically
#     r = int(255 * prob)
#     g = int(255 * (1 - prob))
#     bar_color = f"rgb({r},{g},0)"

#     gauge_fig = go.Figure(
#         go.Indicator(
#             mode="gauge+number",
#             value=prob * 100,
#             title={'text': "Overall Stroke Risk (%)"},
#             gauge={
#                 'axis': {'range': [0, 100], 'ticksuffix': '%'},
#                 'bar': {'color': bar_color},
#                 'steps': [
#                     {'range': [0, 50], 'color': 'green'},
#                     {'range': [50, 100], 'color': 'red'}
#                 ]
#             }
#         )
#     )
#     gauge_fig.update_layout(
#         template="plotly_white",
#         margin=dict(t=40, b=0, l=0, r=0),
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)"
#     )
#     st.plotly_chart(gauge_fig, use_container_width=True)

#     # Navigation
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("🔙 Back to Assessment"):
#             st.switch_page("pages/Risk_Assessment.py")
#     with col2:
#         if st.button("📘 Recommendations"):
#             st.switch_page("pages/Recommendations.py")

# else:
#     st.warning("No input data found. Please complete the Risk Assessment first.")

# # Footer
# st.markdown("""
#   <style>
#     .custom-footer { background: rgba(76,157,112,0.6); color: white; padding: 30px 0;
#                      border-radius: 12px; margin-top: 40px; text-align: center; font-size: 14px; }
#     .custom-footer a { color: white; text-decoration: none; margin: 0 15px; }
#     .custom-footer a:hover { text-decoration: underline; }
#   </style>
#   <div class='custom-footer'>
#     <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
#     <p>
#       <a href='/Home'>Home</a>
#       <a href='/Risk_Assessment'>Risk Assessment</a>
#       <a href='/Results'>Results</a>
#       <a href='/Recommendations'>Recommendations</a>
#     </p>
#     <p style='font-size:12px; margin-top:10px;'>Developed by Victoria Mends</p>
#   </div>
# """, unsafe_allow_html=True)






# import streamlit as st
# import os, joblib, numpy as np, pandas as pd, shap, plotly.graph_objects as go
# from sklearn.preprocessing import StandardScaler

# # ───────────────────────── Page config & CSS ─────────────────────────
# st.set_page_config(page_title="Stroke Risk Results", layout="wide")
# st.markdown("""
#   <style>
#     #MainMenu, footer, header { visibility: hidden; }
#     [data-testid=\"stSidebar\"], [data-testid=\"collapsedControl\"] { display: none; }
#   </style>
# """, unsafe_allow_html=True)

# # ───────────────────────── Title & Navbar ────────────────────────────
# st.title("📊 Stroke Risk Results")
# st.markdown("""
#   <style>
#     .custom-nav { background: #e8f5e9; padding: 15px 0; border-radius: 10px;
#                  display: flex; justify-content: center; gap: 60px; margin-bottom: 30px;
#                  font-size: 18px; font-weight: 600; }
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

# # ─────────────────── Load scaler & GB model ─────────────────────
# @st.cache_resource
# def load_scaler_and_model():
#     base = os.path.dirname(__file__)
#     scaler_path = os.path.join(base, "scaler.pkl")
#     model_path  = os.path.join(base, "best_gb_model.pkl")
#     if not os.path.exists(scaler_path) or not os.path.exists(model_path):
#         st.error("⚠️ Missing scaler.pkl or best_gb_model.pkl in app directory.")
#         st.stop()
#     scaler = joblib.load(scaler_path)
#     model  = joblib.load(model_path)
#     return scaler, model

# scaler, model = load_scaler_and_model()

# # ─────────────────── SHAP Explainer Setup ─────────────────────
# @st.cache_resource
# def get_explainer(_model):
#     background = np.zeros((1, 11))
#     return shap.KernelExplainer(_model.predict_proba, background)

# explainer = get_explainer(model)

# # ─────────────────────────── Main ────────────────────────────
# if "user_data" in st.session_state:
#     ud = st.session_state.user_data
#     # Raw features in training order
#     raw = np.array([[
#         {"Yes":1,"No":0}[ud["heart_disease"]],
#         {"Yes":1,"No":0}[ud["hypertension"]],
#         {"Yes":1,"No":0}[ud["ever_married"]],
#         {"never smoked":1,"formerly smoked":0,"smokes":2}[ud["smoking_status"]],
#         {"Private":2,"Self-employed":3,"Govt_job":0,"Never_worked":1}[ud["work_type"]],
#         {"Male":1,"Female":0}[ud["gender"]],
#         ud["age"], ud["avg_glucose_level"],
#         ud["age"]**2,
#         ud["age"] * ud["avg_glucose_level"],
#         ud["avg_glucose_level"]**2
#     ]], dtype=float)
#     # Scale and predict
#     features = scaler.transform(raw)
#     probs = model.predict_proba(features)[0]
#     pred  = model.predict(features)[0]
#     pos_idx = list(model.classes_).index(1)
#     prob = probs[pos_idx]
#     # Display
#     if pred == 1:
#         st.error(f"⚠️ High risk of stroke.\n\n**Probability:** {prob:.2%}")
#     else:
#         st.success(f"✅ Low risk of stroke.\n\n**Probability:** {prob:.2%}")
#     # SHAP contributions
#     sv = explainer.shap_values(features, nsamples=100)
#     shap_vals = np.array(sv[1] if isinstance(sv, list) else sv).reshape(-1)
#     abs_vals = np.abs(shap_vals[:8])
#     rel_pct  = abs_vals / abs_vals.sum() * 100
#     y_vals   = rel_pct.tolist()
#     feats    = [
#         "Heart Disease","Hypertension","Ever Married",
#         "Smoking Status","Work Type","Gender","Age","Avg Glucose"
#     ]
#     colors   = ["brown","gold","steelblue","purple"]
#     fig_bar = go.Figure(go.Bar(
#         x=feats, y=y_vals,
#         marker=dict(color=[colors[i%4] for i in range(len(feats))]),
#         text=[f"{v:.1f}%" for v in y_vals], textposition="auto",
#         hovertemplate="<b>%{x}</b><br>Contribution: %{y:.1f}%<extra></extra>"
#     ))
#     fig_bar.update_layout(
#         title="Relative Feature Contributions to Stroke Risk",
#         yaxis_title="Contribution (%)", xaxis_tickangle=-45,
#         margin=dict(t=60,b=120)
#     )
#     st.plotly_chart(fig_bar, use_container_width=True)
#     # Gauge chart
#     fig_gauge = go.Figure(go.Indicator(
#         mode="gauge+number", value=prob*100,
#         title={'text':"Overall Stroke Risk (%)"},
#         gauge={'axis':{'range':[0,100]}, 'bar':{'color':'steelblue'},
#                'steps':[{'range':[0,50],'color':'lightgreen'},{'range':[50,100],'color':'lightcoral'}],
#                'threshold':{'line':{'color':'red','width':4},'value':50}}
#     ))
#     fig_gauge.update_layout(margin=dict(t=50,b=0,l=0,r=0))
#     st.plotly_chart(fig_gauge, use_container_width=True)
#     # Navigation
#     c1, c2 = st.columns(2)
#     with c1:
#         if st.button("🔙 Back to Risk Assessment"): 
#             st.switch_page("pages/Risk_Assessment.py")
#     with c2:
#         if st.button("📘 Go to Recommendations"): 
#             st.switch_page("pages/Recommendations.py")
# else:
#     st.warning("Please complete the Risk Assessment first.")

# # Footer
# st.markdown("""
#   <style>
#     .custom-footer { background: rgba(76,157,112,0.6); color: white; padding: 30px 0;
#                      border-radius: 12px; margin-top: 40px; text-align: center; font-size: 14px; }
#     .custom-footer a { color: white; text-decoration: none; margin: 0 15px; }
#     .custom-footer a:hover { text-decoration: underline; }
#   </style>
#   <div class='custom-footer'>
#     <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
#     <p>
#       <a href='/Home'>Home</a>
#       <a href='/Risk_Assessment'>Risk Assessment</a>
#       <a href='/Results'>Results</a>
#       <a href='/Recommendations'>Recommendations</a>
#     </p>
#     <p style='font-size:12px; margin-top:10px;'>Developed by Victoria Mends</p>
#   </div>
# """, unsafe_allow_html=True)










# import streamlit as st
# import os, joblib, shap, plotly.graph_objects as go, pandas as pd, numpy as np
# # ensures engineer_feats is importable

# # ----- place these THREE lines at the *very top* of the page -------------
# import feutils, __main__
# if not hasattr(__main__, "engineer_feats"):
#     setattr(__main__, "engineer_feats", feutils.engineer_feats)
# # -------------------------------------------------------------------------


# # ───────────────────────── Page config & CSS ─────────────────────────
# st.set_page_config(page_title="Stroke Risk Results", layout="wide")
# st.markdown("""
#   <style>
#     #MainMenu, footer, header{visibility:hidden;}
#     [data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none;}
#   </style>""", unsafe_allow_html=True)

# # ───────────────────────── Title & Navbar ────────────────────────────
# st.title("📊 Stroke Risk Results")
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
#   </div>""", unsafe_allow_html=True)

# # ─────────────────────── Load pipeline & explainer ──────────────────
# @st.cache_resource
# def load_pipeline():
#     import feutils, __main__
#     setattr(__main__, "engineer_feats", feutils.engineer_feats)

#     path = os.path.join(os.path.dirname(__file__), "stroke_stacking_pipeline.pkl")
#     return joblib.load(path)


# model = load_pipeline()

# @st.cache_resource
# def get_explainer(_m):
#     """Use TreeExplainer if possible; else fall back to permutation."""
#     try:
#         return shap.TreeExplainer(_m)
#     except shap.utils._exceptions.InvalidModelError:
#         return shap.Explainer(_m, algorithm="permutation")

# explainer = get_explainer(model)

# # ─────────────────────────── Main section ───────────────────────────
# if {"user_data", "prediction_prob"} <= st.session_state.keys():
#     ud        = st.session_state.user_data
#     prob_raw  = float(st.session_state.prediction_prob)
#     pct_disp  = round(prob_raw * 100, 2)

#     st.header("🧠 Stroke Percentage Risk")
#     st.write(f"Based on your inputs, your estimated risk is **{pct_disp:.2f}%**")
#     st.warning("⚠️ Higher Risk of Stroke Detected") if prob_raw > 0.5 else \
#         st.success("✔️ Lower Risk of Stroke Detected")

#     X_df = pd.DataFrame([ud])

#     # ── SHAP contributions ─────────────────────────────────────────
#     shap_vals  = explainer(X_df)
#     base_names = list(ud.keys())          # 8 original fields
#     palette    = ["#A52A2A","#FFD700","#4682B4","#800080"]
#     colors     = [palette[i % 4] for i in range(len(base_names))]

#     if pct_disp == 0:
#         contrib = np.zeros(len(base_names))
#     else:
#         raw     = shap_vals.values[0][:len(base_names)]
#         contrib = np.abs(raw) / np.abs(raw).sum() * prob_raw

#     fig = go.Figure(go.Bar(
#         x=base_names,
#         y=contrib * 100,
#         marker=dict(color=colors),
#         text=[f"{v*100:.2f}%" for v in contrib],
#         textposition="auto",
#         hovertemplate="<b>%{x}</b><br>Contribution: %{y:.2f}%<extra></extra>"
#     ))
#     fig.update_layout(
#         title="How Each Input Contributed to Your Total Risk",
#         yaxis=dict(title="Contribution to Risk (%)", rangemode="tozero"),
#         xaxis=dict(tickangle=-45),
#         margin=dict(t=60, b=120)
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     # Navigation buttons
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("🔙 Back to Risk Assessment"):
#             st.switch_page("pages/Risk_Assessment.py")
#     with col2:
#         if st.button("📘 Go to Recommendations"):
#             st.switch_page("pages/Recommendations.py")
# else:
#     st.warning("No input data found. Please complete the Risk Assessment first.")

# # ─────────────────────────── Footer ────────────────────────────────
# st.markdown("""
#   <style>
#     .custom-footer{background:rgba(76,157,112,0.6);color:white;padding:30px 0;
#                    border-radius:12px;margin-top:40px;text-align:center;font-size:14px;}
#     .custom-footer a{color:white;text-decoration:none;margin:0 15px;}
#     .custom-footer a:hover{text-decoration:underline;}
#   </style>
#   <div class='custom-footer'>
#     <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
#     <p>
#       <a href='/Home'>Home</a><a href='/Risk_Assessment'>Risk Assessment</a>
#       <a href='/Results'>Results</a><a href='/Recommendations'>Recommendations</a>
#     </p>
#     <p style='font-size:12px;margin-top:10px;'>Developed by Victoria Mends</p>
#   </div>""", unsafe_allow_html=True)





# import streamlit as st
# import os, joblib, numpy as np, shap, plotly.graph_objects as go

# # ───────────────────────── Page config & CSS ─────────────────────────
# _ = st.set_page_config(page_title="Stroke Risk Results", layout="wide")
# _ = st.markdown("""
#   <style>
#     #MainMenu, footer, header{visibility:hidden;}
#     [data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none;}
#   </style>
# """, unsafe_allow_html=True)

# # ───────────────────────── Title & Navbar ────────────────────────────
# _ = st.title("📊 Stroke Risk Results")
# _ = st.markdown("""
#   <style>
#     .custom-nav{background:#e8f5e9;padding:15px 0;border-radius:10px;
#                 display:flex;justify-content:center;gap:60px;margin-bottom:30px;
#                 font-size:18px;font-weight:600;}
#     .custom-nav a{text-decoration:none;color:#4C9D70;}
#     .custom-nav a:hover{color:#388e3c;text-decoration:underline;}
#   </style>
#   <div class="custom-nav">
#     <a href='/Home'>Home</a>
#     <a href='/Risk_Assessment'>Risk Assessment</a>
#     <a href='/Results'>Results</a>
#     <a href='/Recommendations'>Recommendations</a>
#   </div>
# """, unsafe_allow_html=True)

# # ──────────────────────── Load model & SHAP explainer ─────────────────────
# @st.cache_resource
# def load_model():
#     path = os.path.join(os.path.dirname(__file__), "best_stacking_model.pkl")
#     if not os.path.exists(path):
#         st.error(f"⚠️ Model file not found at `{path}`"); st.stop()
#     return joblib.load(path)

# model = load_model()

# @st.cache_resource
# def get_explainer(_model):     # leading underscore → skip hashing
#     return shap.TreeExplainer(_model)

# explainer = get_explainer(model)

# # ─────────────────────────── Main section ────────────────────────────
# required_keys = {"user_data", "prediction_prob"}
# if required_keys <= st.session_state.keys():
#     ud        = st.session_state.user_data
#     prob_raw  = float(st.session_state.prediction_prob)     # already computed in previous page
#     pct_disp  = round(prob_raw * 100, 2)                    # two-decimal %

#     _ = st.header("🧠 Stroke Percentage Risk")
#     _ = st.write(f"Based on your inputs, your estimated risk is **{pct_disp:.2f}%**")
#     _ = st.warning("⚠️ Higher Risk of Stroke Detected") if prob_raw > 0.5 else \
#         st.success("✔️ Lower Risk of Stroke Detected")

#     # ── Build feature vector for SHAP (same order as training) ────────────
#     age, glu = ud["age"], ud["avg_glucose_level"]
#     X = np.array([[
#         {"Yes": 1, "No": 0}[ud["heart_disease"]],
#         {"Yes": 1, "No": 0}[ud["hypertension"]],
#         {"Yes": 1, "No": 0}[ud["ever_married"]],
#         {"never smoked": 0, "formerly smoked": 1, "smokes": 2}[ud["smoking_status"]],
#         {"Private": 0, "Self-employed": 1, "Govt_job": 2, "Never_worked": 3}[ud["work_type"]],
#         {"Male": 0, "Female": 1}[ud["gender"]],
#         age, glu, age**2, age * glu, glu**2
#     ]])

#     # ── SHAP contributions ───────────────────────────────────────────────
#     feature_names = ["Heart Disease", "Hypertension", "Ever Married",
#                      "Smoking Status", "Work Type", "Gender", "Age", "Avg Glucose"]
#     palette = ["#A52A2A", "#FFD700", "#4682B4", "#800080"]   # brown, gold, steel-blue, purple
#     colors  = [palette[i % 4] for i in range(8)]

#     if pct_disp == 0.00:                                    # flatten bars if rounded 0 %
#         contrib = np.zeros(len(feature_names))
#     else:
#         sv         = explainer.shap_values(X)
#         shap_vals  = sv[1][0] if isinstance(sv, list) else sv[0]
#         abs8       = np.abs(shap_vals[:8])
#         contrib    = abs8 / abs8.sum() * prob_raw

#     # ── Plotly bar chart ─────────────────────────────────────────────────
#     fig = go.Figure(go.Bar(
#         x=feature_names,
#         y=contrib * 100,
#         marker=dict(color=colors),
#         text=[f"{v*100:.2f}%" for v in contrib],
#         textposition="auto",
#         hovertemplate="<b>%{x}</b><br>Contribution: %{y:.2f}%<extra></extra>"
#     ))
#     fig.update_layout(
#         title="How Each Input Contributed to Your Total Risk",
#         yaxis=dict(title="Contribution to Risk (%)", rangemode="tozero"),
#         xaxis=dict(tickangle=-45),
#         margin=dict(t=60, b=120)
#     )
#     _ = st.plotly_chart(fig, use_container_width=True)

#     # ── Navigation buttons ──────────────────────────────────────────────
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("🔙 Back to Risk Assessment"):
#             st.switch_page("pages/Risk_Assessment.py")
#     with col2:
#         if st.button("📘 Go to Recommendations"):
#             st.switch_page("pages/Recommendations.py")
# else:
#     _ = st.warning("No input data found. Please complete the Risk Assessment first.")

# # ─────────────────────────── Footer ────────────────────────────────
# _ = st.markdown("""
#   <style>
#     .custom-footer{background:rgba(76,157,112,0.6);color:white;padding:30px 0;
#                    border-radius:12px;margin-top:40px;text-align:center;font-size:14px;}
#     .custom-footer a{color:white;text-decoration:none;margin:0 15px;}
#     .custom-footer a:hover{text-decoration:underline;}
#   </style>
#   <div class='custom-footer'>
#     <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
#     <p>
#       <a href='/Home'>Home</a><a href='/Risk_Assessment'>Risk Assessment</a>
#       <a href='/Results'>Results</a><a href='/Recommendations'>Recommendations</a>
#     </p>
#     <p style='font-size:12px;margin-top:10px;'>Developed by Victoria Mends</p>
#   </div>
# """, unsafe_allow_html=True)






