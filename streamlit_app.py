import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import sklearn
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
"""
# ML Project - Student Performance score prediction

Anggota:
- Bryan Mulia
- Jasson Widiarta
- Kasimirus Derryl Odja
- Irving Masahiro
"""


def random_button_callback():
    input_df =pd.read_csv('student-mat.csv', sep=';', usecols=['school','sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
        'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
        'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
        'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health'])
    predict(input_df.sample())

def main_render():
    school = st.selectbox("School:", {"Gabriel Pereira":"GP", "Moushinho da Silveira": "MS"})
    sex = st.selectbox("Sex:", {"Female":"F", "Male": "M"})
    age = st.slider("Age", 6,  24)
    address = st.selectbox("Address:", {"Urban":"U", "Rural": "R"})
    famsize = st.selectbox("Family Size:", {"Less or equal to 3":"LE3", "Greater than 3": "GT3"})
    Pstatus = st.selectbox("Parent's cohabitation status:", {"Less or equal to 3":"LE3", "Greater than 3": "GT3"})
    st.button('random predict', on_click=random_button_callback)

def predict(input_df):
    if 'model' not in st.session_state:
        st.text("model is not loaded")
    else:
        pred = st.session_state.model.predict(input_df)
        st.text('G3: '+str(pred))
        
def main():
    main_render()
    if 'model' not in st.session_state:
        st.session_state.model = pickle.load(open('ridge_model.pkl', 'rb')) 
    else: 
        print("model already loaded")

    return 0

if __name__ == '__main__':
    main()
