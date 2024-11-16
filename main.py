from flask import Flask, render_template, request
import streamlit as st
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  # Untuk menangani typo

app = Flask(__name__)

# Fungsi untuk membaca data dari CSV
def read_csv(file_path):
    # Cek jika file tidak ada
    if not os.path.exists(file_path):
        print(f"File CSV tidak ditemukan di path: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path, sep=';')  # Menggunakan separator ';'
        return df
    except Exception as e:
        print(f"Terjadi kesalahan saat membaca file CSV: {e}")
        return None

# Fungsi untuk mencocokkan merek dan memberikan rekomendasi
def check_brand(brand_name, df):
    if len(brand_name) < 2:
        return "Brand tidak ditemukan.", None, None, None, None

    # Menggunakan TF-IDF untuk menghitung kesamaan
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Nama Brand'])
    query_vec = vectorizer.transform([brand_name])
    
    # Menghitung cosine similarity
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Mencari hasil terbaik
    best_match_index = similarity.argmax()
    best_score = similarity[best_match_index]

    if best_score >= 0.7:  # Threshold untuk kesamaan
        match = df.iloc[best_match_index]
        status = match['Status']
        category = match['Kategori']
        
        # Mencari rekomendasi hanya jika status adalah "boikot"
        recommendations = None
        if status.lower() == 'boikot':
            recommendations = df[(df['Status'] == "Tidak") & (df['Kategori'] == category)]['Nama Brand'].tolist()
        
        # Menentukan warna status
        status_color = 'boikot' if status.lower() == 'boikot' else 'aman'
        
        return f"{match['Nama Brand']}", status, recommendations, status_color, None
    else:
        # Fuzzy matching untuk menangani typo
        possible_match = process.extractOne(brand_name, df['Nama Brand'])
        
        if possible_match:
            matched_brand = possible_match[0]
            return "Brand tidak ditemukan.", None, None, None, matched_brand
        else:
            return "Brand tidak ditemukan.", None, None, None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    status = None
    recommendations = None
    status_color = None
    possible_match = None

    if request.method == 'POST':
        brand_name = request.form['brand_name']
        brands_data = read_csv('static/Produk.csv')
        
        # Cek jika CSV tidak ada atau tidak valid
        if brands_data is None:
            result = "File CSV tidak valid atau tidak ditemukan."
        else:
            result, status, recommendations, status_color, possible_match = check_brand(brand_name, brands_data)

    return render_template('index.html', 
                           result=result, 
                           status=status, 
                           recommendations=recommendations, 
                           status_color=status_color, 
                           possible_match=possible_match)

# Hapus bagian app.run(debug=True) untuk produksi
if __name__ == '__main__':
    app.run()