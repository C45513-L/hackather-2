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
# import os
# from transformers import GenerationConfig
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer



pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Adjust the path to your Tesseract installation
st.set_page_config(layout="wide")

# Load Hugging Face GPT model and tokenizer
model_name = "facebook/opt-1.3b"  # or use a different model like "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

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


# def extract_entities(text):
#     model_name = "bert-base-uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     input_text=text+" Act as an expert in building, real estate, construction industry, what is the type of this document"
#     encoded_text = tokenizer(input_text, truncation=True, padding=True,return_tensors="pt")
#     predictions = model(encoded_text)
#     predicted_label = predictions[0][0]
    
#     return predicted_label

def analyze_document(text):
    prompt = (
        "You are an expert in the building, real estate, and construction industry. "
        "Analyze the following document to identify its type and extract key-value pairs. "
        "The document can be one of the following types: utility bills, drawings, maintenance records, quality assurance reports, contracts, inspection reports, or other types of documents related to the building industry. "
        "Based on the text extracted from the submitted file, identify the type of document and provide key-value pairs if applicable.\n\n"
        "Document Text:\n"
        f"{text}\n\n"
        "Please provide your analysis in the following format:\n"
        "1. Document Type: [Type of the document]\n"
        "2. Key-Value Pairs (if any):\n"
        "   - Key: Value\n"
        "   - Key: Value\n"
        "   - Key: Value\n"

        
    )
    
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=2048, truncation=True)
    outputs = model.generate(inputs, max_length=2048, num_return_sequences=1, no_repeat_ngram_size=2)
    analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return analysis

#----------------------------------------------------------------------------------------------------------------
# VARIABLES
#----------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------
# MAIN
#----------------------------------------------------------------------------------------------------------------

st.title('Building Intel Buddy')
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

        for image in images:
            col1.image(image)
        
        entities = analyze_document(text)
        col1.text_area('Extracted Text', entities, height=1000)
        
    elif file_type.startswith('image/'):
        st.write('Extracting text from Image...')
        text = extract_text_from_image(submit)
        col1, col2 = st.columns(2)
        col1.image(submit)
        # Extract and display entities
        entities = analyze_document(text)
        col2.text_area('Extracted Text', entities, height=1000)


    else:
        st.error('Unsupported file type. Please upload a PDF or image file.')
    
    






