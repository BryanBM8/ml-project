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
- Joel Wilson Suwanto
- Irving Masahiro Samin

Repository: [Github](https://github.com/Jasson9/ml-project)
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
    Medu = st.selectbox("Mother's education level:", {"None":0, "Primary Education 4th grade": 1, "5th to 9th grade": 2, "Secondary Education": 3, "Higher Education": 4})
    Fedu = st.selectbox("Father's education level:", {"None":0, "Primary Education 4th grade": 1, "5th to 9th grade": 2, "Secondary Education": 3, "Higher Education": 4})
    Mjob = st.selectbox("Mother's job:", {"Teacher":"teacher", "Healthcare":"health", "Civil Services":"services", "At Home":"at_home", "Other":"other"})
    Fjob = st.selectbox("Father's job:", {"Teacher":"teacher", "Healthcare":"health", "Civil Services":"services", "At Home":"at_home", "Other":"other"})
    reason = st.selectbox("Reason to choose this school:", {"Close to home":"home", "Reputation":"reputation", "Course preference":"course", "Other":"other"})
    guardian = st.selectbox("Guardian:", {"Mother":"mother", "Father":"father", "Other":"other"})
    traveltime = st.selectbox("Travel time to school:", {"<15 min":1, "15-30 min":2, "30-60 min":3, ">60 min":4})
    studytime = st.selectbox("Weekly study time:", {"<2 hours":1, "2-5 hours":2, "5-10 hours":3, ">10 hours":4})
    failures = st.selectbox("Number of past class failures:", {0:0, 1:1, 2:2, 3:3})
    schoolsup = st.selectbox("Extra educational support:", {"Yes":"yes", "No":"no"})
    famsup = st.selectbox("Family educational support:", {"Yes":"yes", "No":"no"})
    paid = st.selectbox("Extra paid classes within the course subject:", {"Yes":"yes", "No":"no"})
    activities = st.selectbox("Extra-curricular activities:", {"Yes":"yes", "No":"no"})
    nursery = st.selectbox("Attended nursery school:", {"Yes":"yes", "No":"no"})
    higher = st.selectbox("Wants to take higher education:", {"Yes":"yes", "No":"no"})
    internet = st.selectbox("Internet access at home:", {"Yes":"yes", "No":"no"})
    romantic = st.selectbox("With a romantic relationship:", {"Yes":"yes", "No":"no"})
    famrel = st.slider("Quality of family relationships", 1, 5)
    freetime = st.slider("Free time after school", 1, 5)
    goout = st.slider("Going out with friends", 1, 5)
    Dalc = st.slider("Workday alcohol consumption", 1, 5)
    Walc = st.slider("Weekend alcohol consumption", 1, 5)
    health = st.slider("Current health status", 1, 5)
    input_df = pd.DataFrame({
        'school': [school],
        'sex' : [sex],
        'age' : [age],
        'address' : [address],
        'famsize' : [famsize],
        'Pstatus' : [Pstatus],
        'Medu' : [Medu],
        'Fedu' : [Fedu],
        'Mjob' : [Mjob],
        'Fjob' : [Fjob],
        'reason' : [reason],
        'guardian' : [guardian],
        'traveltime' : [traveltime],
        'studytime' : [studytime],
        'failures' : [failures],
        'schoolsup' : [schoolsup],
        'famsup' : [famsup],
        'paid' : [paid],
        'activities' : [activities],
        'nursery' : [nursery],
        'higher' : [higher],
        'internet' : [internet],
        'romantic' : [romantic],
        'famrel' : [famrel],
        'freetime' : [freetime],
        'goout' : [goout],
        'Dalc' : [Dalc],
        'Walc' : [Walc],
        'health' : [health]
    })
    st.button('predict', on_click=lambda: predict(input_df))
    st.button('random predict', on_click=random_button_callback)
    if 'result' in st.session_state:
        st.write("prediction result",st.session_state.result)

def predict(input_df):
    if 'model' not in st.session_state:
        st.session_state.result = "model is not loaded"
    else:
        pred = st.session_state.model.predict(input_df)
        st.session_state.result = str(pred[0])
        
def main():
    main_render()
    if 'model' not in st.session_state:
        st.session_state.model = pickle.load(open('ridge_model.pkl', 'rb')) 
    else: 
        print("model already loaded")

    return 0

if __name__ == '__main__':
    main()
