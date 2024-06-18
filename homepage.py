import streamlit as st
import pandas as pd

def home():
    st.markdown(
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
    )

    st.markdown("""
        <style>
            .main-header {
                font-size: 32px;
                font-weight: bold;
                color: #2c3e50;
                margin-top: 20px;
                padding-bottom: 0px;
            }
            .custom-subheader {
                font-size: 20px;
                padding-top: 0;
                margin-bottom: 30px;
            }
            .section-header{
                font-size: 22px;
            }
            .content{
                font-size: 16px;
                margin-bottom:25px
            }
                
            .image-container{
                display: flex;
                padding: 20px 15px 20px 15px;
                flex-direction: columns;
                margin-bottom: 7px;
                width: 100%;
                height: 300px;
                justify-content: space-between;
            }

                .image-box{
                    height: 100%;
                    width: 45%;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center
                }

                    .image{
                        height: 90%;
                        width: 100%;
                        border: 2px solid black
                    }
                
                    .image-caption{
                        height: 10%;
                        width: 100%;
                        display: flex;
                        justify-content: space-around;
                        align-items: center;
                        margin-top: 5px;
                        font-weight: bold;
                        font-size: 15px
                    }
                
                    .detail{
                        height: 60%;
                        width: 70%;
                        border: 4px solid grey
                    }
                
                    .detail-2{
                        height: 35%;
                        width: 90%;
                        border: 4px solid grey
                    }
                
                    .detail-caption{
                        height: 8%;
                        width: 100%;
                        display: flex;
                        justify-content: space-around;
                        align-items: center;
                        font-weight: bold;
                        font-size: 15px
                    }
                
            .section-header-2{
                margin-top: 18px;
                font-size: 22px;
            }
                
            .section-content-2{
                font-size: 16px;
                margin-bottom: 12px
            }
            
            .flowchart-box{
                width: 150px;
                height: 550px;
                margin-left: 80px;
            }
                
        </style>
        """, unsafe_allow_html=True)
    

    st.markdown('<div class="main-header">EduPred Score</div>', unsafe_allow_html=True)
    st.markdown(
            """
            <div class="custom-subheader">
                Sebuah aplikasi Machine Learning sederhana yang dapat memprediksi score murid berdasarkan beberapa features yang saling berkolerasi
            </div>
            """, unsafe_allow_html=True
        )
    

    st.markdown('<div class="section-header">Apa itu tujuan EduPred Score?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="content">
        Melalui penerapan konsep regression dalam machine learning, EduPred Score bertujuan untuk memberikan sebuah prediction dan insight bagi user mengenai edukasi mereka
    </div>
    """, unsafe_allow_html=True)


    st.markdown('<div class="section-header">Models yang digunakan</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://i.imgur.com/FZDA4jF.png" alt="Gradient Boosting">
                <div class="image-caption">Gradient Boosting</div>
        </div>
        <div class="image-box">
                <img class="detail" src="https://i.imgur.com/d1J5E33.png" alt="Accuracy Gradient Boosting"> 
                <div class="detail-caption">Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://i.imgur.com/EVXw1Jx.png" alt="Random Forest">
                <div class="image-caption">Random Forest</div>
        </div>
        <div class="image-box">
                <img class="detail-2" src="https://i.imgur.com/WbQ5O0j.png alt="Accuracy Random Forest"> 
                <div class="detail-caption">Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://i.imgur.com/id6hCcY.png" alt="Support Vector Machines">
                <div class="image-caption">Support Vector Machines</div>
        </div>
        <div class="image-box">
                <img class="detail-2" src="https://i.imgur.com/gNgXMoN.png alt="Accuracy Support Vector Machines"> 
                <div class="detail-caption">Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)



    df = pd.read_csv('student-mat.csv',sep=';')
    st.markdown('<div class="section-header-2">Student Performance Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Dataset yang kami gunakan berasal dari archive.ics.uci.edu dan terdiri dari atas 32 kolom yang berisi 649 baris data
    </div>
    """, unsafe_allow_html=True)

    styled_df = df.style.set_properties(**{
    'background-color': '#f9f9f9',
    'color': '#333',
    'border-color': '#ddd',
    'text-align': 'center',
    }).set_table_styles(
        [{'selector': 'th', 'props': [('background-color', '#f2f2f2'), ('font-weight', 'bold')]}]
    )

    st.dataframe(styled_df)



    st.markdown('<div class="section-header-2">Flowchart</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        The workflow of how our application is created
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <img class="flowchart-box" src="https://i.imgur.com/9LtMj8g.png" alt="Flowchart"> 
    """, unsafe_allow_html=True)


    return