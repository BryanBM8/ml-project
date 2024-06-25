import streamlit as st
from homepage import home
from eda_page import eda
from preprocessing_page import preprocessing
from training_page import training
from predict_evaluation_page import predict_evaluation
from streamlit_option_menu import option_menu

def router(key,page="Home"):
    match page or st.session_state[key]:
        case "Home":
            home()
        case "EDA":
            eda()
        case "Preprocessing":
            preprocessing()
        case "Training":
            training()
        case "Predict & Evaluation":
            predict_evaluation()
            

def show_menu():
    with st.sidebar:
        menu = option_menu("EduPred Score", ["Home","EDA", "Preprocessing", "Training & Evaluation", "Predict"],
            icons=['house', 'list-task', 'gear', 'cloud-upload'], menu_icon="cast", key="menu", default_index=0)
    router("menu",menu)    