import streamlit as st

# ── Page configuration & styling ─────────────────────────────────────────────
st.set_page_config(page_title="Stroke Risk Recommendations", layout="wide")
st.markdown("""
    <style>
        #MainMenu, header {visibility: hidden;}
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
    </style>
""", unsafe_allow_html=True)

st.title("💡 Stroke Prevention Recommendations")

# ── Custom navbar ────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .custom-nav {
            background-color: #e8f5e9;
            padding: 15px 0;
            border-radius: 10px;
            display: flex;
            justify-content: center;
            gap: 60px;
            margin-bottom: 30px;
            font-size: 18px;
            font-weight: 600;
        }
        .custom-nav a {
            text-decoration: none;
            color: #4C9D70;
        }
        .custom-nav a:hover {
            color: #388e3c;
            text-decoration: underline;
        }
    </style>
    <div class="custom-nav">
        <a href='/Home' target='_self'>Home</a>
        <a href='/Risk_Assessment' target='_self'>Risk Assessment</a>
        <a href='/Results' target='_self'>Results</a>
        <a href='/Recommendations' target='_self'>Recommendations</a>
    </div>
""", unsafe_allow_html=True)

# ── Retrieve risk score from session ──────────────────────────────────────────
risk_prob = st.session_state.get("prediction_prob")
if risk_prob is None:
    st.warning("⚠️ No stroke risk score found. Please complete the assessment first.")
    # Redirect user to input their data
    st.page_link("pages/Risk_Assessment.py", label="Go to Risk Assessment")
    st.stop()

# Convert to percentage
risk_score = risk_prob * 100
st.markdown(f"### 🧠 Your estimated stroke risk is **{risk_score:.2f}%**.")

# ── Personalized recommendations ──────────────────────────────────────────────
st.subheader("🎯 Personalized Recommendations")

# Special case: zero risk
if risk_score == 0:
    st.balloons()
    st.success("🎉 Incredible! Your calculated stroke risk is 0.00%. Keep up these great habits!")
    st.markdown("""
    - Continue your current healthy lifestyle habits.
    - Maintain balanced diet, regular exercise, and stress management.
    - Keep up routine health checkups to stay on track.
    """)

elif risk_score < 30:
    st.success("✅ You have a low risk. Keep up the good work!")
    st.markdown("""
    - Keep up with regular health checkups.
    - Maintain a balanced diet and exercise.
    - Avoid smoking and manage stress effectively.
    """)

elif risk_score < 70:
    st.warning("⚠️ You are at moderate risk. Take proactive steps to lower it.")
    st.markdown("""
    - Monitor and manage blood pressure and glucose levels.
    - Limit alcohol intake and avoid smoking.
    - Consider lifestyle modifications like increasing physical activity.
    """)

else:
    st.error("🚨 You are at high risk. Please take immediate action.")
    st.markdown("""
    - Seek medical advice for detailed cardiovascular assessment.
    - Take prescribed medications if necessary (e.g., antihypertensives).
    - Adopt a strict healthy diet and consistent physical activity routine.
    - Completely avoid tobacco products and excessive alcohol.
    """)

# ── General tips ──────────────────────────────────────────────────────────────
st.subheader("📌 General Stroke Prevention Tips")
st.markdown("""
- Discuss these results with your healthcare provider.
- Develop a personalised prevention plan.
- Schedule regular monitoring of risk factors.
- Learn the warning signs of stroke (**F.A.S.T**) and what to do in case of an emergency.
- Eat more fruits and vegetables, reduce salt and processed foods.
- Maintain a healthy weight and sleep schedule.
- Monitor chronic conditions like diabetes or hypertension.
""")

# ── Navigation buttons ─────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    # Use file path for Risk_Assessment page
    st.page_link("pages/Risk_Assessment.py", label="🔁 Reassess Risk")
with col2:
    # Link back to main app
    st.page_link("app.py", label="🏠 Back to Home")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .custom-footer {
            background-color: rgba(76, 157, 112, 0.6);
            color: white;
            padding: 30px 0;
            border-radius: 12px;
            margin-top: 40px;
            text-align: center;
            font-size: 14px;
            width: 100%;
        }
        .custom-footer a {
            color: white;
            text-decoration: none;
            margin: 0 15px;
        }
        .custom-footer a:hover {
            text-decoration: underline;
        }
    </style>
    <div class="custom-footer">
        <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
        <p>
            <a href='/Home'>Home</a>
            <a href='/Risk_Assessment'>Risk Assessment</a>
            <a href='/Results'>Results</a>
            <a href='/Recommendations'>Recommendations</a>
        </p>
        <p style="font-size:12px; margin-top:10px;">Developed by Victoria Mends</p>
    </div>
""", unsafe_allow_html=True)
