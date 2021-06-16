import streamlit as st
import pandas as pd
import sklearn
import PyPDF2
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import fitz





st.header ("""
Let's Match Your CV with The Job
""")

st.write('Created by : Nezar Abdilah Prakasa.')
st.write('Mathod : Sklearn - Cosine Similarity.')
st.write('This Apps can be pdf extractor text.')

st.header('Masukan File Pdf Job Desk')
uploaded_pdf_JD = st.file_uploader("Silahkan Masukan File Job Desc: ", type=['pdf'])

if uploaded_pdf_JD is not None:
    doc = fitz.open(stream=uploaded_pdf_JD.read(), filetype="pdf")
    text_JD = ""
    for page in doc:
        text_JD += page.getText()
    st.write(text_JD) 
    doc.close()



st.header('Masukan File Pdf CV')

uploaded_pdf = st.file_uploader("Silahkan Masukan File CV: ", type=['pdf'])

if uploaded_pdf is not None:
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    text_cv = ""
    for page in doc:
        text_cv += page.getText()
    st.write(text_cv) 
    doc.close()


st.header("Hasil Match Kamu")

Matcher=[text_JD,text_cv]

cove=CountVectorizer()
count_matrix=cove.fit_transform(Matcher)

MatchPercentage=cosine_similarity(count_matrix)[0][1]*100
MatchPercentage=round(MatchPercentage,2)
st.write('Kecocokan Jobdesk dengan CV mu yaitu sebesar' ,str(MatchPercentage),'%')
