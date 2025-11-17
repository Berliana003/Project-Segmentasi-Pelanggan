import pandas as pd
import numpy as np

# =============================
#  Load Dataset
# =============================
df = pd.read_csv("online_retail_II.csv")

# =============================
#  Tampilkan 5 Data Pertama
# =============================
print("=== 5 Data Pertama ===")
print(df.head())

# =============================
#  Informasi Struktur Dataset
# =============================
print("\n=== Informasi Dataset ===")
print(df.info())

# =============================
#  Deskripsi Data Numerik
# =============================
print("\n=== Deskripsi Fitur Numerik ===")
print(df.describe())

# =============================
#  Cek Jumlah Records & Fitur
# =============================
jumlah_records = len(df)
jumlah_fitur = len(df.columns)

print(f"\nJumlah Data (records) : {jumlah_records}")
print(f"Jumlah Fitur (kolom) : {jumlah_fitur}")

# =============================
#  Cek Tipe Data Tiap Kolom
# =============================
print("\n=== Tipe Data Setiap Kolom ===")
print(df.dtypes)

# =============================
#  Cek Missing Values
# =============================
print("\n=== Missing Values ===")
print(df.isnull().sum())
