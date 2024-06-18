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
                <img class="image" src="https://i.imgur.com/U4hnmCQ.png" alt="Gradient Boosting">
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
                <img class="image" src="https://i.imgur.com/a4HNwDM.png" alt="Random Forest">
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
                <img class="image" src="https://i.imgur.com/gZbfURp.png" alt="Support Vector Machines">
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

    # styled_df = df.style.set_properties(**{
    # 'background-color': '#f9f9f9',
    # 'color': '#333',
    # 'border-color': '#ddd',
    # 'text-align': 'center',
    # }).set_table_styles(
    #     [{'selector': 'th', 'props': [('background-color', '#f2f2f2'), ('font-weight', 'bold')]}]
    # )

    st.dataframe(df)


    st.markdown('<div class="section-header-2">Feature Selection</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Dari semua fitur yang tersedia di dataset, kami memutuskan untuk mengunakan 13 labels saja untuk training model. Features tersebut kami pilih berdasarkan korelasi yang paling berdampak terhadap feature yang ingin kita predict (education score seorang anak)
    </div>
    """, unsafe_allow_html=True)

    df.drop(columns=['schoolsup','health','Dalc','Walc','famsup','freetime','nursery','romantic','Mjob','Fjob','school','studytime','famsize','famrel','absences','activities','Pstatus','guardian','reason'],inplace=True)
    st.dataframe(df)

    st.markdown('<div class="section-header-2">Preprocessing</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Setelah kami drop columns yang tidak akan kami gunakan. Kami akan melakukan tahap-tahap preprocessing seperti removing outliers dan label encoding untuk categorical features agar menjadi numerical features
    </div>
    """, unsafe_allow_html=True)


    result = df.select_dtypes(include='number')
    for i in result.columns:
        percentile25 = df[i].quantile(0.25)
        percentile75 = df[i].quantile(0.75)
        
        iqr = percentile75-percentile25
        
        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr
        
        df = df[df[i] < upper_limit ]
        df = df[df[i] > lower_limit ]
    
    # col=['paid', 'higher', 'internet']
    # dic={'no':0,'yes':1}

    # for i in col:
    #     df[i]=df[i].map(dic)

    # from sklearn import preprocessing
    # le = preprocessing.LabelEncoder()
    # df.address=le.fit_transform(df.address)
    # df.sex=le.fit_transform(df.sex)
    # df.paid=le.fit_transform(df.paid)
    # df.higher=le.fit_transform(df.higher)
    # df.internet=le.fit_transform(df.internet)

    st.dataframe(df)





    st.markdown('<div class="section-header-2">Flowchart</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Berikut merupakan ringkasan dari tahap-tahap yang kita lakukan untuk membuat model machine learning aplikasi kami.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <img class="flowchart-box" src="https://i.imgur.com/9LtMj8g.png" alt="Flowchart"> 
    """, unsafe_allow_html=True)





    return