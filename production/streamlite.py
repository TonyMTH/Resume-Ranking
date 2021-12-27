import streamlit as st
import tkinter as tk
from tkinter import filedialog

# Text/Title
from feature_extraction import FeatureExtraction
from predict import Rank
from utils import Utils

st.title("Query Documents\n\n")

# Set up tkinter
root = tk.Tk()
root.withdraw()
# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)

# Folder picker button
st.write('Please select files folder:')
if st.button('Select'):
    dirname = filedialog.askdirectory(master=root)
    with open("./data/resume_location.txt", "w") as text_file:
        text_file.write(dirname)

query = st.text_area("Enter Query Here")
option = st.selectbox('Which model?', ('TF-DIF', 'BM25', 'NN', 'RandomForest', 'SVM'))
retrieval = st.number_input('How much retrieval?', min_value=1, max_value=None, value=5, step=1)

if st.button('Search'):

    utils = Utils()
    featureExtraction = FeatureExtraction()
    rank = Rank()

    tfidf, bm25, nn, rfr, svm = rank.predict(query)
    model = {'TF-DIF': tfidf, 'BM25': bm25, 'NN': nn, 'RandomForest': rfr, 'SVM': svm}

    i = 1
    for q, _ in model[option]:
        st.write(q.split('\\')[-1])
        if i == retrieval:
            break
        i += 1
