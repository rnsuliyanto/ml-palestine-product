import streamlit as st
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  # Untuk menangani typo

# Fungsi untuk membaca data dari CSV
def read_csv(file_path):
    if not os.path.exists(file_path):
        st.error(f"File CSV tidak ditemukan di path: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path, sep=';')  # Menggunakan separator ';'
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
        return None

# Fungsi untuk mencocokkan merek dan memberikan rekomendasi
def check_brand(brand_name, df):
    if len(brand_name) < 2:
        return "Brand tidak ditemukan.", None, None, None, None

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Nama Brand'])
    query_vec = vectorizer.transform([brand_name])
    
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_match_index = similarity.argmax()
    best_score = similarity[best_match_index]

    if best_score >= 0.7:
        match = df.iloc[best_match_index]
        status = match['Status']
        category = match['Kategori']
        
        recommendations = None
        if status.lower() == 'boikot':
            recommendations = df[(df['Status'] == "Tidak") & (df['Kategori'] == category)]['Nama Brand'].tolist()
        
        status_color = 'boikot' if status.lower() == 'boikot' else 'aman'
        return f"{match['Nama Brand']}", status, recommendations, status_color, None
    else:
        possible_match = process.extractOne(brand_name, df['Nama Brand'])
        matched_brand = possible_match[0] if possible_match else None
        return "Brand tidak ditemukan.", None, None, None, matched_brand

# Streamlit app
st.title("Brand Recommendation App")

brand_name = st.text_input("Masukkan Nama Brand:")
if st.button("Cari"):
    brands_data = read_csv('static/Produk.csv')
    
    if brands_data is not None:
        result, status, recommendations, status_color, possible_match = check_brand(brand_name, brands_data)
        
        st.write(result)
        if status:
            st.write(f"Status: {status}")
        if recommendations:
            st.write(f"Rekomendasi: {recommendations}")
        if possible_match:
            st.write(f"Did you mean: {possible_match}")