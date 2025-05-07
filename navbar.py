import streamlit as st

def show_navbar():
    """Function to display the top navigation bar."""
    st.markdown("""
    <style>
    .navbar {
        background-color: #4CAF50;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .navbar a {
        color: white;
        padding: 14px 20px;
        text-decoration: none;
        font-size: 18px;
    }
    .navbar a:hover {
        background-color: #45a049;
        border-radius: 5px;
    }
    </style>
    <div class="navbar">
        <a href="app.py">Home</a>
        <a href="pages/Risk_Assessment.py">Risk Assessment</a>
        <a href="pages/Results.py">Results</a>
        <a href="pages/Recommendations.py">Recommendations</a>
    </div>
    """, unsafe_allow_html=True)
