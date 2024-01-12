#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


# Membaca dataset
url = "C:/Users/KNRP/Downloads/caesarian.csv"
df = pd.read_csv(url)


# In[3]:


# Menampilkan nama-nama kolom
print(df.columns)


# In[4]:


# Memilih fitur yang relevan
X = df[['Usia', 'Kelahiran_ke-', 'Waktu_Kelahiran', 'Tekanan_darah', 'Kelainan_jantung']]
y = df['Caesarian']


# In[5]:


# Memisahkan data menjadi data pelatihan dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


# Normalisasi fitur menggunakan StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[11]:


# Membuat model KNN dengan k=5 dan menyesuaikan bobot kelas
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')  

# Menggunakan bobot jarak
knn.fit(X_train_scaled, y_train)


# In[12]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)


# In[13]:


# a. Ibu hamil dengan Usia 25 Tahun, Kelahiran ke-1, Waktu kelahiran sesuai HPL, Tekanan darah Normal
new_data = [[25, 1, 0, 1, 0]]
new_data_scaled = scaler.transform(new_data)
result = knn.predict(new_data_scaled)
print("Hasil KNN untuk kondisi ibu hamil dengan Usia 25 Tahun:", result[0])


# In[15]:


new_data = [[35, 1, 0, 2, 0]]
new_data_scaled = scaler.transform(new_data)
result = knn.predict(new_data_scaled)

print("Hasil KNN untuk kondisi ibu hamil dengan Usia 35 Tahun:", result[0])

