import streamlit as st
import pandas as pd
import pickle
from streamlit import  session_state as state

def load_model():
    if 'model' not in st.session_state:
        with open('model.pkl', 'rb') as mo:
            st.session_state.model = pickle.load(mo)

def predict(input_df):
    pred = st.session_state.model.predict(input_df)
    return pred[0]

sex_options = {"Female":"F", "Male": "M"}
address_options = {"Urban":"U", "Rural": "R"}
Medu_options = {"None":0, "Primary Education 4th grade": 1, "5th to 9th grade": 2, "Secondary Education": 3, "Higher Education": 4}
Fedu_options = Medu_options
traveltime_options = {"<15 min":1, "15-30 min":2, "30-60 min":3, ">60 min":4}

def predict_evaluation():
    st.title("Predict & Evaluation")

    if 'preprocessing_done' not in st.session_state or st.session_state.preprocessing_done == False:
        st.warning("Please complete the preprocessing step first.")
        
    if 'training_done' not in st.session_state or st.session_state.training_done == False:
        st.warning("Please complete the training step first.")
        return
    
    sex = st.selectbox("Sex :", sex_options.keys())
    age = st.slider("Age", 6,  24)
    address = st.selectbox("Address:", address_options.keys())
    Medu = st.selectbox("Mother's education level:", Medu_options.keys())
    
    Fedu = st.selectbox("Father's education level:", Fedu_options.keys())

    traveltime = st.selectbox("Travel time to school:", traveltime_options.keys())
    
    failures = st.selectbox("Number of past class failures:", {0, 1, 2, 3})

    paid = 1 if st.selectbox("Extra paid classes within the course subject:", {"Yes", "No"}) == "Yes" else 0

    higher = 1 if st.selectbox("Wants to take higher education:", {"Yes", "No"}) == "Yes" else 0

    internet = 1 if st.selectbox("Internet access at home:", {"Yes", "No"}) == "Yes" else 0

    goout = st.slider("Going out with friends", 1, 5)
    G1 = st.number_input("First period grade", 0, 20)
    G2 = st.number_input("Second period grade", 0, 20)

    print(paid,higher,internet)
    input_df = pd.DataFrame({
        'sex' : [sex_options[sex]],
        'age' : [age],
        'address' : [address_options[address]],
        'Medu' : [Medu_options[Medu]],
        'Fedu' : [Fedu_options[Fedu]],
        'traveltime' : [traveltime_options[traveltime]],
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
        for column in ['address','sex','paid','higher','internet']:
            input_df[column]=state.encoder[column].transform(input_df[column].astype(str))
        
        num_columns = ['age', 'Medu', 'Fedu', 'traveltime','failures','goout', 'G1', 'G2']
        result = input_df[num_columns]
        scaled_df = state.scaler.transform(result)

        input_df = input_df.copy()
        input_df[result.columns] = scaled_df
        st.write(input_df)
        load_model()
        pred=predict(input_df)
        st.write("Predicted result: ", pred)

    return