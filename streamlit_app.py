import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import sklearn
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
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
model_option = st.sidebar.selectbox("Model:", {"Random Forest":"RandomForest.pkl", "SVM":"SVM.pkl", "Gradient Boosting":"XGB.pkl"})
scaler_option = st.sidebar.selectbox("Scaler Option:", {"Standard Scaler":"standard.pkl", "Robust Scaler":"robust.pkl", "MinMax Scaler":"minmax.pkl"})
    

def random_button_callback():
    input_df =pd.read_csv('student-mat.csv', sep=';', usecols=['sex', 'age', 'address', 'Medu', 'Fedu', 
     'traveltime', 'failures', 'paid', 'higher', 'internet','goout', 'G1', 'G2'])
    predict(input_df.sample(),model_options[model_option], scaler_options[scaler_option])

scaler_options ={"Standard Scaler":"standard.pkl", "Robust Scaler":"robust.pkl", "MinMax Scaler":"minmax.pkl"}
model_options = {"Random Forest":"RandomForest.pkl", "SVM":"SVM.pkl", "Gradient Boosting":"gradient_boosting_model.pkl"}
sex_options = {"Female":"F", "Male": "M"}
address_options = {"Urban":"U", "Rural": "R"}
Medu_options = {"None":0, "Primary Education 4th grade": 1, "5th to 9th grade": 2, "Secondary Education": 3, "Higher Education": 4}
Fedu_options = {"None":0, "Primary Education 4th grade": 1, "5th to 9th grade": 2, "Secondary Education": 3, "Higher Education": 4}
traveltime_options = {"<15 min":1, "15-30 min":2, "30-60 min":3, ">60 min":4}
failures_options = {0:0, 1:1, 2:2, 3:3}
paid_options = {"Yes":"yes", "No":"no"}
higher_options = {"Yes":"yes", "No":"no"}
internet_options = {"Yes":"yes", "No":"no"}

def main_render():
    sex = st.sidebar.selectbox("Sex:", {"Female":"F", "Male": "M"})
    age = st.sidebar.slider("Age", 6,  24)
    address = st.sidebar.selectbox("Address:", {"Urban":"U", "Rural": "R"})
    Medu = st.sidebar.selectbox("Mother's education level:", {"None":0, "Primary Education 4th grade": 1, "5th to 9th grade": 2, "Secondary Education": 3, "Higher Education": 4})
    Fedu = st.sidebar.selectbox("Father's education level:", {"None":0, "Primary Education 4th grade": 1, "5th to 9th grade": 2, "Secondary Education": 3, "Higher Education": 4})
    traveltime = st.sidebar.selectbox("Travel time to school:", {"<15 min":1, "15-30 min":2, "30-60 min":3, ">60 min":4})
    failures = st.sidebar.selectbox("Number of past class failures:", {0:0, 1:1, 2:2, 3:3})
    paid = st.sidebar.selectbox("Extra paid classes within the course subject:", {"Yes":"yes", "No":"no"})
    higher = st.sidebar.selectbox("Wants to take higher education:", {"Yes":"yes", "No":"no"})
    internet = st.sidebar.selectbox("Internet access at home:", {"Yes":"yes", "No":"no"})
    goout = st.sidebar.slider("Going out with friends", 1, 5)
    G1 = st.sidebar.number_input("G1")
    G2 = st.sidebar.number_input("G2")
    input_df = pd.DataFrame({
        'sex' : [sex_options[sex]],
        'age' : [age],
        'address' : [address_options[address]],
        'Medu' : [Medu_options[Medu]],
        'Fedu' : [Fedu_options[Fedu]],
        'traveltime' : [traveltime_options[traveltime]],
        'failures' : [failures_options[failures]],
        'paid' : [paid_options[paid]],
        'higher' : [higher_options[higher]],
        'internet' : [internet_options[internet]],
        'goout' : [goout],
        'G1' : [G1],
        'G2' : [G2]
    })
    st.sidebar.button('predict', on_click=lambda: predict(input_df,model_options[model_option], scaler_options[scaler_option]))
    st.sidebar.button('random predict', on_click=random_button_callback)
    if 'result' in st.session_state:
        st.write("prediction result: ",st.session_state.result)

def predict(input_df,model_option,scaler_option):
    print(model_option, scaler_option)
    if 'model_option' not in st.session_state or st.session_state.model_option != model_option:
        try:
            st.session_state.model = pickle.load(open("Models/"+model_option, 'rb'))
        except:
            st.session_state.model = joblib.load("Models/"+model_option)
        st.session_state.model_option = model_option
    if 'scaler_option' not in st.session_state or st.session_state.scaler_option != scaler_option:
        st.session_state.scaler = pickle.load(open("Scaler/"+scaler_option, 'rb'))
        st.session_state.scaler_option = scaler_option
    print(input_df)
    encoder = LabelEncoder()
    original_df = pd.read_csv('student-mat.csv', sep=';', usecols=['sex', 'age', 'address', 'Medu', 'Fedu', 
     'traveltime', 'failures', 'paid', 'higher', 'internet','goout', 'G1', 'G2'])
    # result = input_df.select_dtypes(include='number')
    input_df.address=encoder.fit(original_df.address).transform(input_df.address)
    input_df.sex=encoder.fit(original_df.sex).fit_transform(input_df.sex)
    input_df.paid=encoder.fit(original_df.paid).fit_transform(input_df.paid)
    input_df.higher=encoder.fit(original_df.higher).fit_transform(input_df.higher)
    input_df.internet=encoder.fit(original_df.internet).fit_transform(input_df.internet)

    result = input_df[['age', 'Medu', 'Fedu', 
     'traveltime', 'failures', 'G1', 'G2']]
    scaled_df = st.session_state.scaler.transform(result)
    input_df = input_df.copy()
    input_df[result.columns] = scaled_df
    print(input_df)
    pred = st.session_state.model.predict(input_df)
    st.session_state.result = str(pred[0])
        
def main():
    main_render()
    return 0

if __name__ == '__main__':
    main()
