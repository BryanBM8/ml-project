import streamlit as st
import pandas as pd

def home():
    st.markdown(
        """
# ML Project - Student Performance score prediction

Anggota:
- 2602054764 - Bryan Mulia
- 2602057646 - Jasson Widiarta
- 2602059960 - Kasimirus Derryl Odja
- 2602077635 - Irving Masahiro Samin
- 2602050860 - Joel Wilson Suwanto

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
                        height: 55%;
                        width: 125%;
                        border: 4px solid grey
                    }
                
                    .detail-caption{
                        height: 8%;
                        margin-right: 2%;
                        width: 125%;
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
                height: 750px;
                margin-left: 100px;
            }
                
            .scaled-container{
                display: flex;
                padding: 20px 15px 20px 15px;
                flex-direction: column;
                margin-bottom: 7px;
                width: 100%;
                height: 300px;
            }
                
            .scaled-caption{
                height: 10%;
                width: 100%;
                display: flex;
                justify-content: space-around;
                align-items: center;
                margin-top: 5px;
                font-weight: bold;
                font-size: 13px
            }
                
            ul {
                list-style-type: disc;
                margin-left: 20px;
                padding-left: 0; 
                font-size: 15px;  
                color: #333333;  
            }
            li {
                margin-bottom: 8px; 
                line-height: 1.5;  
            
            .inner-li {
                list-style-type: square;
                font-size: 15px;
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
    <ul>
        <li>Gradient Boost</li>
        <li>Random Forest</li>
        <li>Support Vector Regressor</li>
    </ul>
    """, unsafe_allow_html=True)
   



    df = pd.read_csv('student-mat.csv',sep=';')
    st.markdown('<div class="section-header-2">Student Performance Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Dataset yang kami gunakan berasal dari archive.ics.uci.edu dan terdiri dari atas 32 kolom yang berisi 649 baris data
    </div>
    """, unsafe_allow_html=True)


    st.dataframe(df)


    st.markdown('<div class="section-header-2">Feature Selection</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Dari semua fitur yang tersedia di dataset, kami memutuskan untuk mengunakan 13 labels saja untuk training model. Features tersebut kami pilih berdasarkan korelasi yang paling berdampak terhadap feature yang ingin kita predict (education score seorang anak)
    </div>
    """, unsafe_allow_html=True)

    df.drop(columns=['schoolsup','health','Dalc','Walc','famsup','freetime','nursery','romantic','Mjob','Fjob','school','studytime','famsize','famrel','absences','activities','Pstatus','guardian','reason'],inplace=True)
    st.dataframe(df)

    st.markdown('<div class="section-header-2">Data Preprocessing</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Setelah kami drop columns yang tidak akan kami gunakan. Kami akan split data menjadi training dan testing sets dan melakukan tahap-tahap preprocessing: removing outliers dan label encoding untuk mengubah categorical features dalam data menjadi numerical features.
    </div>
    """, unsafe_allow_html=True)


    result = df.select_dtypes(include='number')
    for i in result.columns:
        percentile25 = df[i].quantile(0.25)
        percentile75 = df[i].quantile(0.75)
        
        iqr = percentile75-percentile25
        
        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr
        
        df = df[(df[i] <= upper_limit) & (df[i] >= lower_limit)]
    
    col=['paid', 'higher', 'internet']
    dic={'no':0,'yes':1}

    for i in col:
        df[i]=df[i].map(dic)

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    df.address=le.fit_transform(df.address)
    df.sex=le.fit_transform(df.sex)
    df.paid=le.fit_transform(df.paid)
    df.higher=le.fit_transform(df.higher)
    df.internet=le.fit_transform(df.internet)

    st.dataframe(df)


    st.markdown('<div class="section-header-2">Data Scaling</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Setelah kami lakukan preprocessing terhadap data. Kami akan mencoba untuk menyari best scaler terhadap data kita. Scaler yang kami akan uji berupa: StandardScaler, MinMaxScaler, dan Robust Scaler. Setelah kami menguji setiap parameter dari ketiga scaler tersebut, kami akan fit best scaler tersebut kepada data kita agar kita bisa menggunakannya untuk normalize numerical data dalam training dan testing sets kita. 
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="scaled-container">
        <img class="image" src="https://i.imgur.com/Ybpn3Wv.png" alt="Scaled Gradient Boost">
        <div class="scaled-caption">Scaled Numerical Data untuk Gradient Boost Model training dan testing (excluding binary mapped data)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="scaled-container">
        <img class="image" src="https://i.imgur.com/6hEW7OU.png" alt="Scaled Random Forest">
        <div class="scaled-caption">Scaled Numerical Data untuk Random Forest Model training dan testing (excluding binary mapped data)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="scaled-container">
        <img class="image" src="https://i.imgur.com/DwVToPV.png" alt="Scaled SVR">
        <div class="scaled-caption">Scaled Numerical Data untuk Support Vector Regression Model training dan testing (excluding binary mapped data)</div>
    </div>
    """, unsafe_allow_html=True)

    gradient_boost_params = [
        ('learning_rate', 0.01),
        ('max_depth', 3),
        ('min_samples_leaf', 4),
        ('min_samples_split', 10),
        ('n_estimators', 300)
    ]
    random_forest_params = [
        ('max_depth', 10),
        ('max_features', 'sqrt'),
        ('min_samples_leaf', 1),
        ('min_samples_split', 2),
        ('n_estimators', 500)
    ]
    svr_params = [
        ('C', 1),
        ('epsilon', 0.1),
        ('gamma', 'scale'),
        ('kernel', 'linear')
    ]

    st.markdown('<div class="section-header">Finding Best Model Parameters</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Berdasarkan scoring metric dari <b>'GridSearchCV'</b> yang mengevaluasikan setiap kombinasi dari parameter model untuk menemukan parameter model terbaik untuk data yang kita miliki. Kita akan menggunakan best parameter yang kita temukan ini untuk train model kita.
    </div>
    """, unsafe_allow_html=True)

    def format_params(params):
        inner_list = []
        for param, value in params:
            inner_list.append(f"<li class='inner-li'><b>{param}</b> : {value}</li>")
        return "<ul>" + "".join(inner_list) + "</ul>"

    st.markdown(f"""
    <ul>
        <li><b>Gradient Boost:</b> {format_params(gradient_boost_params)}</li>
        <li><b>Random Forest:</b> {format_params(random_forest_params)}</li>
        <li><b>Support Vector Regressor:</b> {format_params(svr_params)}</li>
    </ul>
    """, unsafe_allow_html=True)








    st.markdown('<div class="section-header-2">Model Training and Evaluation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Berikut merupakan hasil dari training model kita beserta dengan performance evaluation dari setiap model tersebut. Model kita di nilai berdasarkan metrics R-Squared, Mean Absolute Error, Mean Squared Error, dan Root Mean Squared Error.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://i.imgur.com/MY1TleS.png" alt="Gradient Boosting">
                <div class="image-caption">Gradient Boosting</div>
        </div>
        <div class="image-box">
                <img class="detail-2" src="https://i.imgur.com/nHa87Gg.png" alt="Accuracy Gradient Boosting"> 
                <div class="detail-caption">Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://i.imgur.com/19EyjRf.png" alt="Random Forest">
                <div class="image-caption">Random Forest</div>
        </div>
        <div class="image-box">
                <img class="detail-2" src="https://i.imgur.com/beQLJZ7.png" alt="Accuracy Random Forest"> 
                <div class="detail-caption">Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="image-container">
        <div class="image-box">
                <img class="image" src="https://i.imgur.com/iYWmWIO.png" alt="Support Vector Regression">
                <div class="image-caption">Support Vector Regression</div>
        </div>
        <div class="image-box">
                <img class="detail-2" src="https://i.imgur.com/TNoMan5.png" alt="Accuracy Support Vector Regression"> 
                <div class="detail-caption">Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)





    st.markdown('<div class="section-header-2">Flowchart</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-content-2">
        Berikut merupakan ringkasan dari tahap-tahap yang kita lakukan untuk membuat model machine learning aplikasi kami.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <img class="flowchart-box" src="https://i.imgur.com/zh3xuea.png" alt="Flowchart"> 
    """, unsafe_allow_html=True)


    


    return