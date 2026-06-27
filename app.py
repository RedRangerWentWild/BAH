import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="ExoDetect — ISRO BAH 2026",
    page_icon="🔭",
    layout="wide"
)

st.markdown("""
<style>
  #MainMenu, header, footer { visibility: hidden; }
  .block-container { padding: 0 !important; max-width: 100% !important; }
  section[data-testid="stSidebar"] { display: none; }
  iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)

with open('dashboard.html', 'r') as f:
    html = f.read()

components.html(html, height=2200, scrolling=True)