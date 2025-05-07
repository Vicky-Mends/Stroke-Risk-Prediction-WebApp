import streamlit as st
from gtts import gTTS
import base64
import os
import numpy as np
import sys, __main__  

# ── Define polynomial helper for unpickling ─────────────────────────────────
def add_poly(X_array):
    age         = X_array[:, 0]
    glu         = X_array[:, 1]
    age_sq      = age ** 2
    interaction = age * glu
    glu_sq      = glu ** 2
    return np.c_[X_array, age_sq, interaction, glu_sq]

# Monkey-patch into __main__ so pipeline can find add_poly
setattr(__main__, "add_poly", add_poly)

# --- make engineer_feats available to every page ---------------------------
def engineer_feats(df):
    df = df.copy()
    df["age_sq"]      = df["age"]**2
    df["glucose_sq"]  = df["avg_glucose_level"]**2
    df["age_glucose"] = df["age"] * df["avg_glucose_level"]
    return df

setattr(__main__, "engineer_feats", engineer_feats)
# ---------------------------------------------------------------------------

# Set page configuration
st.set_page_config(page_title="Stroke Risk Prediction", layout="wide")

# Encode image as Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        b64_encoded = base64.b64encode(img_file.read()).decode()
        return f"data:image/png;base64,{b64_encoded}"

encoded_image0 = get_base64_image("strokeprediction.png")  # original hero
encoded_image1 = get_base64_image("image2.png")           # first new slide
encoded_image2 = get_base64_image("image3.png")           # second new slide

# Hide default Streamlit elements
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
</style>
""", unsafe_allow_html=True)

# Custom Header
st.markdown("""
<div style="background-color: #4C9D70; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
    <h1 style="color: white; text-align: center; margin: 0;">🧠 Stroke Risk Assessment Tool</h1>
    <p style="color: white; text-align: center; font-size: 18px;">Empowering you to take control of your brain health</p>
</div>
""", unsafe_allow_html=True)

# Custom Navbar
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
    }
</style>
<div class="custom-nav">
    <a href='/Home' target='_self'>Home</a>
    <a href='/Risk_Assessment' target='_self'>Risk Assessment</a>
    <a href='/Results' target='_self'>Results</a>
    <a href='/Recommendations' target='_self'>Recommendations</a>
</div>
""", unsafe_allow_html=True)

# Hero Banner with slideshow
st.markdown(f"""
<style>
    .hero-banner {{
        position: relative;
        width: 100%;
        height: 650px;
        border-radius: 15px;
        overflow: hidden;
        margin-bottom: 30px;
    }}
    .hero-banner .slides img {{
        position: absolute;
        top: 0; left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        opacity: 0;
        animation: slideAnim 12s infinite;
    }}
    .hero-banner .slides img:nth-child(1) {{ animation-delay: 0s; }}
    .hero-banner .slides img:nth-child(2) {{ animation-delay: 4s; }}
    .hero-banner .slides img:nth-child(3) {{ animation-delay: 8s; }}

    @keyframes slideAnim {{
        0% {{ opacity: 1; }}
        33.33% {{ opacity: 1; }}
        33.34% {{ opacity: 0; }}
        100% {{ opacity: 0; }}
    }}

    .hero-text-overlay {{
        position: absolute;
        bottom: 30px;
        left: 40px;
        color: white;
        background-color: rgba(0, 0, 0, 0.45);
        padding: 20px;
        border-radius: 10px;
    }}
    .hero-text-overlay h2 {{ margin: 0; font-size: 28px; }}
    .hero-text-overlay p {{ margin-top: 5px; font-size: 16px; }}
</style>
<div class="hero-banner">
    <div class="slides">
        <img src="{encoded_image0}" alt="Slide 1">
        <img src="{encoded_image1}" alt="Slide 2">
        <img src="{encoded_image2}" alt="Slide 3">
    </div>
    <div class="hero-text-overlay">
        <h2>Early Detection Saves Lives</h2>
        <p>Explore stroke prevention strategies with our intelligent tool</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Call to Action
st.markdown("""
<div style='background-color:#e6f2ff; padding:30px; border-radius:12px; text-align:center;'>
    <h4>📝 Assess Your Stroke Risk</h4>
    <p>Click below to use our intelligent tool and evaluate your risk level.</p>
    <a href='/Risk_Assessment' target='_self'><button style='background-color:#4C9D70; color:white; padding:12px 24px; font-size:16px; border:none; border-radius:8px; cursor:pointer; transition:all 0.3s ease;'>➡️ Start Risk Assessment</button></a>
</div>
<style>
button:hover { background-color:#3e8e41; transform:scale(1.05); }
</style>
""", unsafe_allow_html=True)

# Intro section
st.title("🧠 Learn About Stroke")
st.markdown("""
<p style='font-size:18px;'>
A stroke happens when the blood supply to part of your brain is interrupted or reduced,
preventing brain tissue from getting oxygen and nutrients. Early detection can save lives.
</p>
""", unsafe_allow_html=True)

# Narration text
full_page_text = """
Assess Your Stroke Risk

Click below to use our intelligent tool and evaluate your risk level

Learn About Stroke

A stroke happens when the blood supply to part of your brain is interrupted or reduced,
preventing brain tissue from getting oxygen and nutrients. Early detection can save lives.

Types of Stroke:
- Ischemic: Blockage in brain arteries.
- Hemorrhagic: Burst blood vessels in the brain.
- TIA: Temporary blockage (mini-stroke).

Common Causes:
- High blood pressure
- Heart disease
- Diabetes
- Smoking
- Obesity and cholesterol

Prevention:
- Control blood pressure and sugar
- Exercise regularly
- Eat a healthy diet
- Stop smoking

Symptoms:
- Sudden numbness or weakness (face, arm, leg)
- Confusion, speech trouble
- Vision problems
- Dizziness or balance issues

Recognize a Stroke (FAST):
- F: Face drooping
- A: Arm weakness
- S: Speech difficulty
- T: Time to call emergency

Stroke Statistics:
- 2nd leading cause of death globally
- 12.2 million cases in 2020
- 5.5 million deaths annually
"""

# Audio narration
def generate_audio(text, filename="full_page.mp3"):
    tts = gTTS(text, lang='en')
    tts.save(filename)
    with open(filename, "rb") as f:
        audio_data = f.read()
        b64_audio = base64.b64encode(audio_data).decode()
    st.markdown("### 🔊 Listen to this page")
    st.markdown(
        f"<audio controls style='width:100%; margin-top:20px;'><source src='data:audio/mp3;base64,{b64_audio}' type='audio/mp3'></audio>",
        unsafe_allow_html=True
    )

generate_audio(full_page_text)

# Info Cards
def info_card(icon, title, content):
    st.markdown(
        f"""
        <div style='background-color:#f0f8ff; padding:25px; border-radius:15px; margin-bottom:20px; box-shadow:0 4px 12px rgba(0,0,0,0.05);'>
            <h4 style='font-size:22px;'>{icon} {title}</h4>
            <div style='font-size:16px;'>{content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

col1, col2 = st.columns(2)
with col1:
    info_card("🧩", "Types of Stroke", """
    <ul>
        <li><strong>Ischemic:</strong> Blockage in brain arteries.</li>
        <li><strong>Hemorrhagic:</strong> Burst blood vessels in the brain.</li>
        <li><strong>TIA:</strong> Temporary blockage (mini-stroke).</li>
    </ul>
    """
    )
    info_card("❗", "Common Causes", """
    <ul>
        <li>High blood pressure</li>
        <li>Heart disease</li>
        <li>Diabetes</li>
        <li>Smoking</li>
        <li>Obesity and cholesterol</li>
    </ul>
    """
    )
    info_card("🏃", "Prevention", """
    <ul>
        <li>Control blood pressure & sugar</li>
        <li>Exercise regularly</li>
        <li>Eat a healthy diet</li>
        <li>Stop smoking</li>
    </ul>
    """
    )
with col2:
    info_card("⚠️", "Symptoms", """
    <ul>
        <li>Sudden numbness or weakness (face, arm, leg)</li>
        <li>Confusion, speech trouble</li>
        <li>Vision problems</li>
        <li>Dizziness or balance issues</li>
    </ul>
    """
    )
    info_card("⏱️", "Recognize a Stroke (FAST)", """
    <strong>Use the FAST test:</strong>
    <ul>
        <li><strong>F:</strong> Face drooping</li>
        <li><strong>A:</strong> Arm weakness</li>
        <li><strong>S:</strong> Speech difficulty</li>
        <li><strong>T:</strong> Time to call emergency</li>
    </ul>
    """
    )
    info_card("📊", "Stroke Statistics", """
    <ul>
        <li>2nd leading cause of death globally</li>
        <li>12.2 million cases in 2020</li>
        <li>5.5 million deaths annually</li>
    </ul>
    """
    )

# Footer
st.markdown("""
<style>
    .custom-footer {
        background-color:rgba(76,157,112,0.6);
        color:white;
        padding:30px 0;
        border-radius:12px;
        margin-top:40px;
        text-align:center;
        font-size:14px;
        width:100%;
    }
    .custom-footer a { color:white; text-decoration:none; margin:0 15px; }
    .custom-footer a:hover { text-decoration:underline; }
    .footer-text { width:80%; margin:0 auto; }
</style>
<div class="custom-footer">
    <div class="footer-text">
        <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
        <p>
            <a href='/Home'>Home</a> <a href='/Risk_Assessment'>Risk Assessment</a>
            <a href='/Results'>Results</a> <a href='/Recommendations'>Recommendations</a>
        </p>
        <p style="font-size:12px; margin-top:10px;">Developed by Victoria Mends</p>
    </div>
</div>
""", unsafe_allow_html=True)








# import streamlit as st
# from gtts import gTTS
# import base64

# import sys, __main__            # NEW

# # --- make engineer_feats available to every page ------------------
# def engineer_feats(df):
#     df = df.copy()
#     df["age_sq"]      = df["age"]**2
#     df["glucose_sq"]  = df["avg_glucose_level"]**2
#     df["age_glucose"] = df["age"] * df["avg_glucose_level"]
#     return df

# import __main__
# setattr(__main__, "engineer_feats", engineer_feats)
# # ------------------------------------------------------------------




# # Set page configuration
# st.set_page_config(page_title="Stroke Risk Prediction", layout="wide")

# # Encode image as Base64
# def get_base64_image(image_path):
#     with open(image_path, "rb") as img_file:
#         b64_encoded = base64.b64encode(img_file.read()).decode()
#         return f"data:image/png;base64,{b64_encoded}"

# # Get Base64 image strings
# encoded_image0 = get_base64_image("strokeprediction.png")  # original hero
# encoded_image1 = get_base64_image("image2.png")           # first new slide
# encoded_image2 = get_base64_image("image3.png")           # second new slide

# # Hide default Streamlit elements
# st.markdown("""
# <style>
#     #MainMenu, footer, header {visibility: hidden;}
#     [data-testid="stSidebar"] {display: none;}
#     [data-testid="collapsedControl"] {display: none;}
# </style>
# """, unsafe_allow_html=True)

# # Custom Header
# st.markdown("""
# <div style="background-color: #4C9D70; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
#     <h1 style="color: white; text-align: center; margin: 0;">🧠 Stroke Risk Assessment Tool</h1>
#     <p style="color: white; text-align: center; font-size: 18px;">Empowering you to take control of your brain health</p>
# </div>
# """, unsafe_allow_html=True)

# # Custom Navbar
# st.markdown("""
# <style>
#     .custom-nav {
#         background-color: #e8f5e9;
#         padding: 15px 0;
#         border-radius: 10px;
#         display: flex;
#         justify-content: center;
#         gap: 60px;
#         margin-bottom: 30px;
#         font-size: 18px;
#         font-weight: 600;
#     }
#     .custom-nav a {
#         text-decoration: none;
#         color: #4C9D70;
#     }
#     .custom-nav a:hover {
#         color: #388e3c;
#         text-decoration: underline;
#     }
# </style>
# <div class="custom-nav">
#     <a href='/Home' target='_self'>Home</a>
#     <a href='/Risk_Assessment' target='_self'>Risk Assessment</a>
#     <a href='/Results' target='_self'>Results</a>
#     <a href='/Recommendations' target='_self'>Recommendations</a>
# </div>
# """, unsafe_allow_html=True)

# # Hero Banner with full cover slideshow
# st.markdown(f"""
# <style>
#     .hero-banner {{
#         position: relative;
#         width: 100%;
#         height: 650px;
#         border-radius: 15px;
#         overflow: hidden;
#         margin-bottom: 30px;
#     }}
#     .hero-banner .slides {{
#         position: relative;
#         width: 100%;
#         height: 100%;
#     }}
#     .hero-banner .slides img {{
#         position: absolute;
#         top: 0; left: 0;
#         width: 100%;
#         height: 100%;
#         object-fit: cover;
#         opacity: 0;
#         animation: slideAnim 12s infinite;
#     }}
#     .hero-banner .slides img:nth-child(1) {{ animation-delay: 0s; }}
#     .hero-banner .slides img:nth-child(2) {{ animation-delay: 4s; }}
#     .hero-banner .slides img:nth-child(3) {{ animation-delay: 8s; }}

#     @keyframes slideAnim {{
#         0% {{ opacity: 1; }}
#         33.33% {{ opacity: 1; }}
#         33.34% {{ opacity: 0; }}
#         100% {{ opacity: 0; }}
#     }}

#     .hero-text-overlay {{
#         position: absolute;
#         bottom: 30px;
#         left: 40px;
#         color: white;
#         background-color: rgba(0, 0, 0, 0.45);
#         padding: 20px;
#         border-radius: 10px;
#     }}
#     .hero-text-overlay h2 {{ margin: 0; font-size: 28px; }}
#     .hero-text-overlay p {{ margin-top: 5px; font-size: 16px; }}
# </style>
# <div class="hero-banner">
#     <div class="slides">
#         <img src="{encoded_image0}" alt="Slide 1">
#         <img src="{encoded_image1}" alt="Slide 2">
#         <img src="{encoded_image2}" alt="Slide 3">
#     </div>
#     <div class="hero-text-overlay">
#         <h2>Early Detection Saves Lives</h2>
#         <p>Explore stroke prevention strategies with our intelligent tool</p>
#     </div>
# </div>
# """, unsafe_allow_html=True)

# # Call to Action
# st.markdown("""
# <div style='background-color:#e6f2ff; padding:30px; border-radius:12px; text-align:center; margin-top:30px;'>
#     <h4>📝 Assess Your Stroke Risk</h4>
#     <p>Click below to use our intelligent tool and evaluate your risk level.</p>
#     <a href='/Risk_Assessment' target='_self'><button style='background-color:#4C9D70; color:white; padding:12px 24px; font-size:16px; border:none; border-radius:8px; cursor:pointer; transition:all 0.3s ease;'>➡️ Start Risk Assessment</button></a>
# </div>
# <style>
# button:hover { background-color:#3e8e41; transform:scale(1.05); }
# </style>
# """, unsafe_allow_html=True)

# # Intro section
# st.title("🧠 Learn About Stroke")
# st.markdown("""
# <p style='font-size:18px;'>
# A stroke happens when the blood supply to part of your brain is interrupted or reduced,
# preventing brain tissue from getting oxygen and nutrients. Early detection can save lives.
# </p>
# """, unsafe_allow_html=True)

# # Narration text
# full_page_text = """
# Assess Your Stroke Risk

# Click below to use our intelligent tool and evaluate your risk level

# Learn About Stroke

# A stroke happens when the blood supply to part of your brain is interrupted or reduced,
# preventing brain tissue from getting oxygen and nutrients. Early detection can save lives.

# Types of Stroke:
# - Ischemic: Blockage in brain arteries.
# - Hemorrhagic: Burst blood vessels in the brain.
# - TIA: Temporary blockage (mini-stroke).

# Common Causes:
# - High blood pressure
# - Heart disease
# - Diabetes
# - Smoking
# - Obesity and cholesterol

# Prevention:
# - Control blood pressure and sugar
# - Exercise regularly
# - Eat a healthy diet
# - Stop smoking

# Symptoms:
# - Sudden numbness or weakness (face, arm, leg)
# - Confusion, speech trouble
# - Vision problems
# - Dizziness or balance issues

# Recognize a Stroke (FAST):
# - F: Face drooping
# - A: Arm weakness
# - S: Speech difficulty
# - T: Time to call emergency

# Stroke Statistics:
# - 2nd leading cause of death globally
# - 12.2 million cases in 2020
# - 5.5 million deaths annually

# """

# # Audio narration
# def generate_audio(text, filename="full_page.mp3"):
#     tts = gTTS(text, lang='en')
#     tts.save(filename)
#     with open(filename, "rb") as f:
#         audio_data = f.read()
#         b64_audio = base64.b64encode(audio_data).decode()
#     st.markdown("### 🔊 Listen to this page")
#     st.markdown(
#         f"<audio controls style='width:100%; margin-top:20px;'><source src='data:audio/mp3;base64,{b64_audio}' type='audio/mp3'></audio>",
#         unsafe_allow_html=True
#     )

# generate_audio(full_page_text)

# # Info Cards
# def info_card(icon, title, content):
#     st.markdown(
#         f"""
#         <div style='background-color:#f0f8ff; padding:25px; border-radius:15px; margin-bottom:20px; box-shadow:0 4px 12px rgba(0,0,0,0.05);'>
#             <h4 style='font-size:22px;'>{icon} {title}</h4>
#             <div style='font-size:16px;'>{content}</div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# col1, col2 = st.columns(2)
# with col1:
#     info_card("🧩", "Types of Stroke", """
#     <ul>
#         <li><strong>Ischemic:</strong> Blockage in brain arteries.</li>
#         <li><strong>Hemorrhagic:</strong> Burst blood vessels in the brain.</li>
#         <li><strong>TIA:</strong> Temporary blockage (mini-stroke).</li>
#     </ul>
#     """)
#     info_card("❗", "Common Causes", """
#     <ul>
#         <li>High blood pressure</li>
#         <li>Heart disease</li>
#         <li>Diabetes</li>
#         <li>Smoking</li>
#         <li>Obesity and cholesterol</li>
#     </ul>
#     """)
#     info_card("🏃", "Prevention", """
#     <ul>
#         <li>Control blood pressure & sugar</li>
#         <li>Exercise regularly</li>
#         <li>Eat a healthy diet</li>
#         <li>Stop smoking</li>
#     </ul>
#     """)
# with col2:
#     info_card("⚠️", "Symptoms", """
#     <ul>
#         <li>Sudden numbness or weakness (face, arm, leg)</li>
#         <li>Confusion, speech trouble</li>
#         <li>Vision problems</li>
#         <li>Dizziness or balance issues</li>
#     </ul>
#     """)
#     info_card("⏱️", "Recognize a Stroke (FAST)", """
#     <strong>Use the FAST test:</strong>
#     <ul>
#         <li><strong>F:</strong> Face drooping</li>
#         <li><strong>A:</strong> Arm weakness</li>
#         <li><strong>S:</strong> Speech difficulty</li>
#         <li><strong>T:</strong> Time to call emergency</li>
#     </ul>
#     """)
#     info_card("📊", "Stroke Statistics", """
#     <ul>
#         <li>2nd leading cause of death globally</li>
#         <li>12.2 million cases in 2020</li>
#         <li>5.5 million deaths annually</li>
#     </ul>
#     """)

# # # Call to Action
# # st.markdown("""
# # <div style='background-color:#e6f2ff; padding:30px; border-radius:12px; text-align:center; margin-top:30px;'>
# #     <h4>📝 Assess Your Stroke Risk</h4>
# #     <p>Click below to use our intelligent tool and evaluate your risk level.</p>
# #     <a href='/Risk_Assessment' target='_self'><button style='background-color:#4C9D70; color:white; padding:12px 24px; font-size:16px; border:none; border-radius:8px; cursor:pointer; transition:all 0.3s ease;'>➡️ Start Risk Assessment</button></a>
# # </div>
# # <style>
# # button:hover { background-color:#3e8e41; transform:scale(1.05); }
# # </style>
# # """, unsafe_allow_html=True)

# # Footer
# st.markdown("""
# <style>
#     .custom-footer {
#         background-color:rgba(76,157,112,0.6);
#         color:white;
#         padding:30px 0;
#         border-radius:12px;
#         margin-top:40px;
#         text-align:center;
#         font-size:14px;
#         width:100%;
#         position:relative;
#     }
#     .custom-footer a { color:white; text-decoration:none; margin:0 15px; }
#     .custom-footer a:hover { text-decoration:underline; }
#     .footer-text { width:80%; margin:0 auto; }
# </style>
# <div class="custom-footer">
#     <div class="footer-text">
#         <p>&copy; 2025 Stroke Risk Assessment Tool | All rights reserved</p>
#         <p>
#             <a href='/Home'>Home</a> <a href='/Risk_Assessment'>Risk Assessment</a>
#             <a href='/Results'>Results</a> <a href='/Recommendations'>Recommendations</a>
#         </p>
#         <p style="font-size:12px; margin-top:10px;">Developed by Victoria Mends</p>
#     </div>
# </div>
# """, unsafe_allow_html=True)
