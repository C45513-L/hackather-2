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
import os

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Adjust the path to your Tesseract installation
st.set_page_config(layout="wide")



#----------------------------------------------------------------------------------------------------------------
# FUNCTION
#----------------------------------------------------------------------------------------------------------------

@st.cache_data
def lambeth_resi_map():
    df = gpd.read_file('./data/Housing_Estates.geojson')
    st.write(df)
    shapefile_path = './data/Housing_Estates.shp'
    gdf = gpd.read_file(shapefile_path) 
    geojson_data = gdf.to_json()
    geojson_dict = json.loads(geojson_data)
    
    # Create a Plotly map using the GeoJSON data
    fig = px.choropleth_mapbox(
        gdf,
        geojson=geojson_dict,
        locations=gdf.index,
        #color="ES",  # Replace with the column you want to use for coloring
        mapbox_style="carto-positron",
        center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
        zoom=10,
        opacity=0.5
    )

    st.plotly_chart(fig, use_container_width=True)


def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name

        # Convert PDF to images
        images = convert_from_path(tmp_file_path)
        for image in images:
            text += pytesseract.image_to_string(image)

        # Remove the temporary file
        os.remove(tmp_file_path)

    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    return text

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error reading image file: {e}")
        text = ""
    return text



#----------------------------------------------------------------------------------------------------------------
# VARIABLES
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# MAIN
#----------------------------------------------------------------------------------------------------------------

st.title('Lambeth Innovator Hackather')
st.write('Welcome to the page, this is a unified platform we collective submit our building data for future measurement')

lambeth_resi_map()

#----------------------------------------------------------------------------------------------------------------
# SIDE BAR
#----------------------------------------------------------------------------------------------------------------

user = st.sidebar.radio('Who are you?', options=['Owner of Unit', 'Owner of Building', 'Contractor', 'Developer'])
st.sidebar.divider()

st.sidebar.subheader('Building Thread Submission')

submit = st.sidebar.file_uploader('Upload your file', label_visibility='hidden')

if submit:
    file_type = submit.type
    if file_type == 'application/pdf':
        st.write('Extracting text from PDF...')
        col1, col2 = st.columns(2)
        text, images = extract_text_from_pdf(submit)
        
        col1.text_area('Extracted Text', text, height=300)
        for image in images:
            col2.image(image)
    elif file_type.startswith('image/'):
        st.write('Extracting text from Image...')
        text = extract_text_from_image(submit)
        col1, col2 = st.columns(2)
        col1.text_area('Extracted Text', text, height=300)
        col2.image(submit)
    else:
        st.error('Unsupported file type. Please upload a PDF or image file.')
    
    










