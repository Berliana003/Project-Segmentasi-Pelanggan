import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from clean_data import clean_data
from exploration import explore_clean_data
from feature_rfm import feature_engineering  

# ==========================================
# EDA FEATURE ENGINEERING
# ==========================================
def eda_feature_engineering(fitur_customer: pd.DataFrame, df_clean: pd.DataFrame):
    print("\n===== EDA FEATURE ENGINEERING =====")

    # Pastikan kolom tersedia
    required_columns = ["Recency_Days", "Total_Transactions", "Total_Spending"]
    if not all(col in fitur_customer.columns for col in required_columns):
        raise ValueError(f"❌ Kolom tidak lengkap! Harus ada: {required_columns}")

    # ------------ 1. Distribusi Recency ------------
    print("\n▶ Distribusi Recency (hari sejak transaksi terakhir):")
    print(fitur_customer["Recency_Days"].describe())

    plt.figure(figsize=(6,4))
    sns.histplot(fitur_customer["Recency_Days"], bins=30, kde=True)
    plt.title("Distribusi Recency (hari)")
    plt.xlabel("Hari sejak transaksi terakhir")
    plt.ylabel("Jumlah Customer")
    plt.show()

    # ------------ 2. Distribusi Frequency ------------
    print("\n▶ Distribusi Frequency (jumlah transaksi unik):")
    print(fitur_customer["Total_Transactions"].describe())

    plt.figure(figsize=(6,4))
    sns.histplot(fitur_customer["Total_Transactions"], bins=30, kde=True)
    plt.title("Distribusi Frequency (Total Transaksi)")
    plt.xlabel("Jumlah Transaksi")
    plt.ylabel("Jumlah Customer")
    plt.show()

    # ------------ 3. Distribusi Monetary ------------
    print("\n▶ Distribusi Monetary (total spending):")
    print(fitur_customer["Total_Spending"].describe())

    plt.figure(figsize=(6,4))
    sns.histplot(fitur_customer["Total_Spending"], bins=30, kde=True)
    plt.title("Distribusi Monetary (Total Spending)")
    plt.xlabel("Total Spending")
    plt.ylabel("Jumlah Customer")
    plt.show()

    # ------------ 4. Negara dengan pelanggan terbanyak ------------
    if "Country" in df_clean.columns and "Customer_ID" in df_clean.columns:
        print("\n▶ Top 10 Negara dengan Customer Terbanyak:")
        negara_customer = df_clean.groupby("Country")["Customer_ID"].nunique().sort_values(ascending=False).head(10)
        print(negara_customer)

        plt.figure(figsize=(10,5))
        sns.barplot(x=negara_customer.index, y=negara_customer.values)
        plt.xticks(rotation=45)
        plt.title("Top 10 Negara dengan Customer Terbanyak")
        plt.xlabel("Negara")
        plt.ylabel("Jumlah Customer")
        plt.show()
    else:
        print("\n⚠ Kolom 'Country' tidak ditemukan!")

    print("\n📌 INSIGHT EDA:")
    print("- Recency tinggi → pelanggan tidak aktif, butuh reaktivasi.")
    print("- Frequency rendah → pelanggan jarang transaksi, perlu strategi retensi.")
    print("- Monetary condong ke sedikit pelanggan → potensial segmentasi VIP.")
    print("- Negara teratas bisa jadi target pasar utama.")


# ==========================================
# MAIN WORKFLOW
# ==========================================
if __name__ == "__main__":
    print("\n🚀 Menjalankan pipeline feature engineering dan EDA...\n")

    # 1. Cleaning dataset
    output_file = clean_data()
    df_clean = pd.read_csv(output_file, parse_dates=["InvoiceDate"])

    # 2. Eksplorasi awal dataset
    explore_clean_data(df_clean)

    # 3. Hitung fitur RFM
    fitur_customer = feature_engineering(df_clean)

    # 4. Rename kolom agar sesuai dengan EDA
    fitur_customer = fitur_customer.rename(columns={
        "Recency": "Recency_Days",
        "Frequency": "Total_Transactions",
        "Monetary": "Total_Spending"
    })

    print("\n🔍 Kolom fitur customer setelah rename:", fitur_customer.columns.tolist())

    # 5. EDA Feature Engineering
    eda_feature_engineering(fitur_customer, df_clean)

    # 6. Simpan hasil fitur
    fitur_customer.to_csv("feature_customer.csv", index=True)
    print("\n✔ Analisis selesai! Data siap untuk clustering.")
    print("📁 Disimpan ke: feature_customer.csv")
