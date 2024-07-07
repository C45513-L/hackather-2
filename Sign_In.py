"""
Team member: Pitnaree, Shafna, Eva, Cassie
Hackther Date: 5th Jul - 7th Jul
"""

#----------------------------------------------------------------------------------------------------------------
# LIBRARY
#----------------------------------------------------------------------------------------------------------------

import streamlit as st 
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
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
from stpyvista import stpyvista

import pyvista as pv
from pyvista import examples


pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Adjust the path to your Tesseract installation

# # Load Hugging Face GPT model and tokenizer
# model_name = "facebook/opt-1.3b"  # or use a different model like "distilgpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

st.set_page_config(
    page_title="Sign Up",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="auto"
)



#----------------------------------------------------------------------------------------------------------------
# FUNCTION
#----------------------------------------------------------------------------------------------------------------

# @st.experimental_dialog("What are you up to today?")
# def action():

#     action = st.radio(label='Activity', options=['Visit Portfolio', 'Add a New Building'])
    
#     if st.button("Enter"):
#         st.session_state.vote = { "action": action}
#         st.rerun()




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
            st.session_state.show_dialog = True
            st.experimental_rerun()  # Rerun the app to reflect the change
        else:
            st.error("Please ensure that all fields are filled correctly and the passwords match.")

@st.experimental_dialog("What are you up to today?")
def show_dialog():
    
    action = st.radio(label='Activity', options=['Visit Portfolio', 'Add a New Building'])
    
    if st.button("Enter"):
        st.session_state.action = {"action": action}
        st.session_state.show_dialog = False
        st.experimental_rerun()

def upload_page():
    if st.session_state.populated:
        st.sidebar.image('./data/image.png')
        st.sidebar.write('')
        st.sidebar.write("### Populated Information")
        st.sidebar.write(f"**Address:** {st.session_state.address}")
        st.sidebar.write(f"**Building Name:** {st.session_state.building_name}")
        st.sidebar.write(f"**Building Type:** {st.session_state.build_type}")
        st.sidebar.write(f"**Floor Plan Level:** {', '.join(st.session_state.build_level)}")

        st.toast("Building Intel has been populated.", icon="🔥")

        st.sidebar.divider()
        record_upload   = st.sidebar.button('Upload Record',use_container_width=True, key='record')
        analytics       = st.sidebar.button("Show Analytics", use_container_width=True, key='analytics')
        history         = st.sidebar.button("History", use_container_width=True, key='history')
        dashboard = st.sidebar.button("Show Analytics", use_container_width=True, key='dashboard')

        st.subheader(st.session_state.building_name)
        st.write('')
        st.write('')
        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                if st.session_state.upload is not None:
                    image = Image.open(st.session_state.upload)
                    st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.container(height=500, border=True)

        if analytics:
            st.subheader('Analytics')

            met1, met2, met3, met4, met5 = st.columns(5)

            met1.metric('No. Level', value=len(st.session_state.build_level))
            met2.metric('No. Unit', value=np.random.randint(1, 10))
            met3.metric('GFA in sqm', value=np.random.randint(1, 1000))
            met4.metric('NIA in sqm', value=np.random.randint(1, 1000))
            met5.metric('EPC Rating', value='C')

            # Historical energy consumption graph
            years = list(range(2010, 2023))
            consumption = np.random.randint(200, 500, size=len(years))
            prediction = np.random.randint(150, 300, size=len(years))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=years, y=consumption, mode='lines+markers', name='Consumption'))
            fig.add_trace(go.Scatter(x=years, y=prediction, mode='lines+markers', name='Prediction'))

            fig.update_layout(title='Historical Building Energy Consumption and Prediction Trends',
                              xaxis_title='Year',
                              yaxis_title='Energy Consumption (kWh)',
                              legend_title='Legend')

            st.plotly_chart(fig)

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
        if not address or not building_name or not build_type or not build_level or not upload:
            st.error("Please fill in all the fields and upload a file.")
        else:
            
            st.session_state.address = address
            st.session_state.building_name = building_name
            st.session_state.build_type = build_type
            st.session_state.build_level = build_level
            st.session_state.upload = upload
            st.session_state.populated = True
            st.experimental_rerun()



#----------------------------------------------------------------------------------------------------------------
# BODY - SIGN UP
#----------------------------------------------------------------------------------------------------------------

# Initialize session state if not already done
if 'signed_up' not in st.session_state:
    st.session_state.signed_up = False

if 'populated' not in st.session_state:
    st.session_state.populated = False

if 'address' not in st.session_state:
    st.session_state.address = ""

if 'building_name' not in st.session_state:
    st.session_state.building_name = ""

if 'build_type' not in st.session_state:
    st.session_state.build_type = ""

if 'build_level' not in st.session_state:
    st.session_state.build_level = []

if 'upload' not in st.session_state:
    st.session_state.upload = None

if 'show_dialog' not in st.session_state:
    st.session_state.show_dialog = False

if 'action' not in st.session_state:
    st.session_state.action = None

# Main App
if st.session_state.show_dialog:
    show_dialog()
elif st.session_state.action and st.session_state.action["action"] == "Add a New Building":
    upload_page()
elif st.session_state.action and st.session_state.action["action"] == "Visit Portfolio":
    st.empty()  # Placeholder for future functionality
else:
    show_sign_up_page()


# Define the file path
stl_file_path = "./data"

reader = pv.STLReader('/Users/cassieleong/Documents/hackather-2/data/building _model_prototype.stl')

mesh = reader.read()
        
plotter = pv.Plotter(window_size=[800, 600])
plotter.add_mesh(mesh,color='white')


## Export to an external pythreejs
model_html = "model.html"
other = plotter.export_html(model_html)

## Read the exported model
with open(model_html,'r') as file: 
    model = file.read()

## Show in webpage
st.components.v1.html(model,height=500)