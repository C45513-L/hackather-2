"""
Team member: Pitnaree, Shafna, Eva, Cassie
Hackther Date: 5th Jul - 7th Jul
"""

#----------------------------------------------------------------------------------------------------------------
# LIBRARY
#----------------------------------------------------------------------------------------------------------------

import streamlit as st 
import pandas as pd
import geopandas as gpd
import plotly.express as px
import json
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import fitz  # PyMuPDF
import pytesseract
import cv2
from PIL import Image
import io
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Adjust the path to your Tesseract installation

# # Load Hugging Face GPT model and tokenizer
# model_name = "facebook/opt-1.3b"  # or use a different model like "distilgpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

st.set_page_config(
    page_title="Sign Up",
    page_icon="👋",
    layout="centered"
)



#----------------------------------------------------------------------------------------------------------------
# FUNCTION
#----------------------------------------------------------------------------------------------------------------
# Function to show the sign-up page
def show_sign_up_page():
    st.title('Sign Up')
    st.write('First create your account')

    name = st.text_input('Full name')
    email = st.text_input('Work Email')
    password = st.text_input('Password', type='password')
    confirm_pass = st.text_input('Confirm your password', type='password')

    st.write('')
    st.write('')

    sign_up_butt = st.button('SIGN UP', use_container_width=True)

    st.write('')
    st.write('')

    st.write('Already have an account? Login')

    if sign_up_butt:
        if password == confirm_pass and name and email:
            st.session_state.signed_up = True
            st.experimental_rerun()  # Rerun the app to reflect the change
        else:
            st.error("Please ensure that all fields are filled correctly and the passwords match.")

# Function to show the upload page
def upload_page():
    if 'populated' in st.session_state and st.session_state.populated:

        st.sidebar.image('./data/image.png')
        st.sidebar.write('')
        st.sidebar.write("### Populated Information")
        st.sidebar.write(f"**Address:** {st.session_state.address}")
        st.sidebar.write(f"**Building Type:** {st.session_state.build_type}")
        st.sidebar.write(f"**Floor Plan Level:** {', '.join(st.session_state.build_level)}")
        st.toast("Building Intel has been populated.", icon="🔥")
        return
    
    logo1, logo2, logo3 = st.columns((0.2, 1, 0.2))
    logo2.image('./data/image.png', width=110)
    logo2.header('Building Intel Partner')

    address = st.text_input('Address', value='Canterbury House, 1 Royal Street')
    building_name = st.text_input('Building Name', value='Canterbury House')
    build_type = st.text_input('Building Type', value='Residential = Ex-Council Estate Flat')
    build_level = st.multiselect(label='Floor Plan Level', options=['Basement', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8', 'Level 9', 'Level 10'])

    st.write('')
    upload = st.file_uploader('file upload', label_visibility='hidden')

    bib = st.button('POPULATE MY BUILDING INTEL', use_container_width=True)

    if bib:
        if not address or not build_type or not build_level or not upload:
            st.error("Please fill in all the fields and upload a file.")
        else:
            st.session_state.address = address
            st.session_state.build_type = build_type
            st.session_state.build_level = build_level
            st.session_state.upload = upload
            st.session_state.populated = True
            st.rerun()




#----------------------------------------------------------------------------------------------------------------
# BODY - SIGN UP
#----------------------------------------------------------------------------------------------------------------

# Initialize session state if not already done
if 'signed_up' not in st.session_state:
    st.session_state.signed_up = False

# Main App
if st.session_state.signed_up:
    upload_page()
else:
    show_sign_up_page()

