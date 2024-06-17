import streamlit as st
import pandas as pd
import pickle

def load_model():
    if 'model' not in st.session_state:
        with open('model.pkl', 'rb') as mo:
            st.session_state.model = pickle.load(mo)

def predict(input_df):
    pred = st.session_state.model.predict(input_df)
    st.session_state.result = str(pred[0])

def predict_evaluation():
    st.title("Predict & Evaluation")

    if 'preprocessing_done' not in st.session_state or st.session_state.preprocessing_done == False:
        st.warning("Please complete the preprocessing step first.")
        
    if 'training_done' not in st.session_state or st.session_state.training_done == False:
        st.warning("Please complete the training step first.")
        return
    
    sex = 0 if st.selectbox("Sex :", ["Female", "Male"]) == "Female" else 1
    age = st.slider("Age", 6,  24)
    address = 0 if st.selectbox("Address:", {"Urban", "Rural"}) == "Rural" else 1

    Medu = st.selectbox("Mother's education level:", {"None", "Primary Education 4th grade", "5th to 9th grade", "Secondary Education", "Higher Education"})
    if Medu=="None":
        Medu=0
    elif Medu=="Primary Education 4th grade":
        Medu=1
    elif Medu=="5th to 9th grade":
        Medu=2
    elif Medu=="Secondary Education":
        Medu=3
    elif Medu=="Higher Education":
        Medu=4
    
    Fedu = st.selectbox("Father's education level:", {"None":0, "Primary Education 4th grade": 1, "5th to 9th grade": 2, "Secondary Education": 3, "Higher Education": 4})
    if Fedu=="None":
        Fedu=0
    elif Fedu=="Primary Education 4th grade":
        Fedu=1
    elif Fedu=="5th to 9th grade":
        Fedu=2
    elif Fedu=="Secondary Education":
        Fedu=3
    elif Fedu=="Higher Education":
        Fedu=4

    traveltime = st.selectbox("Travel time to school:", {"<15 min":1, "15-30 min":2, "30-60 min":3, ">60 min":4})
    if traveltime=="<15 min":
        traveltime=1
    elif traveltime=="15-30 min":
        traveltime=2
    elif traveltime=="30-60 min":
        traveltime=3
    elif traveltime==">60 min":
        traveltime=4
    
    failures = st.selectbox("Number of past class failures:", {0, 1, 2, 3})

    paid = 1 if st.selectbox("Extra paid classes within the course subject:", {"Yes", "No"}) == "Yes" else 0

    higher = 1 if st.selectbox("Wants to take higher education:", {"Yes", "No"}) == "Yes" else 0

    internet = 1 if st.selectbox("Internet access at home:", {"Yes", "No"}) == "Yes" else 0

    goout = st.slider("Going out with friends", 1, 5)
    G1 = st.number_input("First period grade", 0, 20)
    G2 = st.number_input("Second period grade", 0, 20)
    input_df = pd.DataFrame({
        'sex' : [sex],
        'age' : [age],
        'address' : [address],
        'Medu' : [Medu],
        'Fedu' : [Fedu],
        'traveltime' : [traveltime],
        'failures' : [failures],
        'paid' : [paid],
        'higher' : [higher],
        'internet' : [internet],
        'goout' : [goout],
        'G1' : [G1],
        'G2' : [G2]
    })
    
    # st.button('predict', on_click=lambda: predict(input_df))
    if st.button('Predict'):
        load_model()
        predict(input_df)
        st.write("Predicted result: ", st.session_state.result)

    return