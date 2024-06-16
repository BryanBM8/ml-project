
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def eda():
    st.title("Exploratory Data Analysis")
    df=pd.read_csv('student-mat.csv',sep=';')
    st.write("Dataframe:")
    st.write(df)
    st.write("Informasi Dataframe:")
    st.write(df.info())

    st.write("Matriks Korelasi")
    numeric_df = df.select_dtypes(include=[float, int])
    correlation_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Matriks Korelasi')
    st.pyplot(fig)
        
    st.write("Distribusi Umur Siswa")
    fig, ax = plt.subplots()
    sns.histplot(df['age'], kde=True, ax=ax)
    ax.set_title('Distribusi Umur Siswa')
    ax.set_xlabel('Umur')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)

    st.write("Distribusi Nilai Akhir")
    fig, ax = plt.subplots()
    sns.histplot(df['G3'], kde=True, ax=ax)
    ax.set_title('Distribusi Nilai Akhir')
    ax.set_xlabel('Nilai Akhir')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)

    st.write("Tren Nilai Ujian (G1, G2, G3)")
    mean_scores = df[['G1', 'G2', 'G3']].mean()
    fig, ax = plt.subplots()
    mean_scores.plot(kind='line', marker='o', ax=ax)
    ax.set_title('Tren Nilai Ujian (G1, G2, G3)')
    ax.set_xlabel('Tahap Ujian')
    ax.set_ylabel('Rata-rata Nilai')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['G1', 'G2', 'G3'])
    st.pyplot(fig)
    
    st.write("Distribusi Jumlah Kegagalan")
    fig, ax = plt.subplots()
    sns.countplot(x='failures', data=df, ax=ax)
    ax.set_title('Distribusi Jumlah Kegagalan')
    ax.set_xlabel('Jumlah Kegagalan')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)

    st.write("Nilai Akhir Berdasarkan Alamat")
    fig, ax = plt.subplots()
    sns.boxplot(x='address', y='G3', data=df, ax=ax)
    ax.set_title('Nilai Akhir Berdasarkan Alamat')
    ax.set_xlabel('Alamat')
    ax.set_ylabel('Nilai Akhir')
    st.pyplot(fig)

    st.write("Nilai Akhir Berdasarkan Ukuran Keluarga")
    fig, ax = plt.subplots()
    sns.boxplot(x='famsize', y='G3', data=df, ax=ax)
    ax.set_title('Nilai Akhir Berdasarkan Ukuran Keluarga')
    ax.set_xlabel('Ukuran Keluarga')
    ax.set_ylabel('Nilai Akhir')
    st.pyplot(fig)


    st.write("Nilai Akhir Berdasarkan Status Orang Tua")
    fig, ax = plt.subplots()
    sns.boxplot(x='Pstatus', y='G3', data=df, ax=ax)
    ax.set_title('Nilai Akhir Berdasarkan Status Orang Tua')
    ax.set_xlabel('Status Orang Tua')
    ax.set_ylabel('Nilai Akhir')
    st.pyplot(fig)

    st.write("Nilai Akhir Berdasarkan Pendidikan Ibu")
    fig, ax = plt.subplots()
    sns.boxplot(x='Medu', y='G3', data=df, ax=ax)
    ax.set_title('Nilai Akhir Berdasarkan Pendidikan Ibu')
    ax.set_xlabel('Pendidikan Ibu')
    ax.set_ylabel('Nilai Akhir')
    st.pyplot(fig)
    
    
    st.write("Nilai Akhir Berdasarkan Pendidikan Ayah")
    fig, ax = plt.subplots()
    sns.boxplot(x='Fedu', y='G3', data=df, ax=ax)
    ax.set_title('Nilai Akhir Berdasarkan Pendidikan Ayah')
    ax.set_xlabel('Pendidikan Ayah')
    ax.set_ylabel('Nilai Akhir')
    st.pyplot(fig)

    st.write("Pengaruh Absensi Terhadap Nilai Akhir")
    fig, ax = plt.subplots()
    sns.scatterplot(x='absences', y='G3', data=df, ax=ax)
    ax.set_title('Pengaruh Absensi Terhadap Nilai Akhir')
    ax.set_xlabel('Absensi')
    ax.set_ylabel('Nilai Akhir')
    st.pyplot(fig)


    st.write("Nilai Akhir Berdasarkan Kursus Tambahan Berbayar")
    fig, ax = plt.subplots()
    sns.boxplot(x='paid', y='G3', data=df, ax=ax)
    ax.set_title('Nilai Akhir Berdasarkan Kursus Tambahan Berbayar')
    ax.set_xlabel('Kursus Tambahan Berbayar')
    ax.set_ylabel('Nilai Akhir')
    st.pyplot(fig)

    st.write("Nilai Akhir Berdasarkan Aktivitas Ekstrakurikuler")
    fig, ax = plt.subplots()
    sns.boxplot(x='activities', y='G3', data=df, ax=ax)
    ax.set_title('Nilai Akhir Berdasarkan Aktivitas Ekstrakurikuler')
    ax.set_xlabel('Aktivitas Ekstrakurikuler')
    ax.set_ylabel('Nilai Akhir')
    st.pyplot(fig)



    st.write("Nilai Akhir Berdasarkan Jumlah Kegagalan")
    fig, ax = plt.subplots()
    sns.boxplot(x='failures', y='G3', data=df, ax=ax)
    ax.set_title('Nilai Akhir Berdasarkan Jumlah Kegagalan')
    ax.set_xlabel('Jumlah Kegagalan')
    ax.set_ylabel('Nilai Akhir')
    st.pyplot(fig)
    

    st.write("Nilai Akhir Berdasarkan Rencana Pendidikan Lebih Tinggi")
    fig, ax = plt.subplots()
    sns.boxplot(x='higher', y='G3', data=df, ax=ax)
    ax.set_title('Nilai Akhir Berdasarkan Rencana Pendidikan Lebih Tinggi')
    ax.set_xlabel('Rencana Pendidikan Lebih Tinggi')
    ax.set_ylabel('Nilai Akhir')
    st.pyplot(fig)
    

    st.write("Nilai Akhir Berdasarkan Akses Internet di Rumah")
    fig, ax = plt.subplots()
    sns.boxplot(x='internet', y='G3', data=df, ax=ax)
    ax.set_title('Nilai Akhir Berdasarkan Akses Internet di Rumah')
    ax.set_xlabel('Akses Internet')
    ax.set_ylabel('Nilai Akhir')
    st.pyplot(fig)
    
    st.write("Nilai Akhir Berdasarkan Hubungan Keluarga")
    fig, ax = plt.subplots()
    sns.boxplot(x='famrel', y='G3', data=df, ax=ax)
    ax.set_title('Nilai Akhir Berdasarkan Hubungan Keluarga')
    ax.set_xlabel('Hubungan Keluarga')
    ax.set_ylabel('Nilai Akhir')
    st.pyplot(fig)
    

    st.write("Nilai Akhir Berdasarkan Aktivitas Keluar")
    fig, ax = plt.subplots()
    sns.boxplot(x='goout', y='G3', data=df, ax=ax)
    ax.set_title('Nilai Akhir Berdasarkan Aktivitas Keluar')
    ax.set_xlabel('Aktivitas Keluar')
    ax.set_ylabel('Nilai Akhir')
    st.pyplot(fig)
    
        
    return
