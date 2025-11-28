import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from clean_data import clean_data
from exploration import explore_clean_data


# ==============================================
# FEATURE ENGINEERING (RFM)
# ==============================================
def feature_engineering(df: pd.DataFrame):
    df = df.copy()  # hindari warning
    print("\n===== FEATURE ENGINEERING PER CUSTOMER =====")

    # Hitung total spending (Monetary)
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    # Tanggal referensi = transaksi terakhir
    reference_date = df["InvoiceDate"].max()

    # Hitung fitur per customer
    fitur_customer = df.groupby("Customer_ID").agg({
        "Invoice": "nunique",      # Frequency
        "TotalPrice": "sum",       # Monetary
        "InvoiceDate": "max"       # Last transaction
    })

    fitur_customer.columns = ["Frequency", "Monetary", "Last_Transaction"]

    # Hitung recency dalam hari
    fitur_customer["Recency"] = (reference_date - fitur_customer["Last_Transaction"]).dt.days

    # Hanya ambil fitur final RFM
    fitur_customer = fitur_customer.drop(columns=["Last_Transaction"])
    fitur_customer = fitur_customer[["Recency", "Frequency", "Monetary"]]

    print("\n▶ Contoh hasil feature engineering (RFM):")
    print(fitur_customer.head())

    print("\n📌 Insight awal:")
    print("- Recency rendah → pelanggan masih aktif.")
    print("- Frequency tinggi → pelanggan sering transaksi.")
    print("- Monetary tinggi → pelanggan berpotensi menjadi VIP.")

    return fitur_customer


# ==============================================
# EDA RFM
# ==============================================
def rfm_exploration(fitur_customer: pd.DataFrame, df_clean: pd.DataFrame):
    print("\n===== EDA RFM (Recency, Frequency, Monetary) =====")

    # 1. Distribusi Recency
    print("\n▶ Distribusi Recency:")
    print(fitur_customer["Recency"].describe())
    plt.figure(figsize=(8, 4))
    sns.histplot(fitur_customer["Recency"], bins=30, kde=True)
    plt.title("Distribusi Recency (hari)")
    plt.show()

    # 2. Distribusi Frequency
    print("\n▶ Distribusi Frequency:")
    print(fitur_customer["Frequency"].describe())
    plt.figure(figsize=(8, 4))
    sns.histplot(fitur_customer["Frequency"], bins=30, kde=True)
    plt.title("Distribusi Frequency")
    plt.show()

    # 3. Distribusi Monetary
    print("\n▶ Distribusi Monetary:")
    print(fitur_customer["Monetary"].describe())
    plt.figure(figsize=(8, 4))
    sns.histplot(fitur_customer["Monetary"], bins=30, kde=True)
    plt.title("Distribusi Monetary (Total Spending)")
    plt.show()

    # 4. Jumlah customer per negara
    if "Country" in df_clean.columns:
        pelanggan_per_negara = df_clean.groupby("Country")["Customer_ID"].nunique().sort_values(ascending=False)
        print("\n▶ Jumlah customer per negara:")
        print(pelanggan_per_negara)

        plt.figure(figsize=(10, 5))
        pelanggan_per_negara.head(10).plot(kind='bar')
        plt.title("Top 10 Negara dengan Customer Terbanyak")
        plt.ylabel("Jumlah Customer Unik")
        plt.show()

    print("\n📌 INSIGHT OTOMATIS:")
    print("- Recency tinggi → banyak pelanggan lama (tidak aktif).")
    print("- Frequency mayoritas rendah → kemungkinan banyak pelanggan hanya beli sekali.")
    print("- Monetary sangat skew → sedikit pelanggan jadi big spender.")
    print("- Negara dengan customer terbanyak → target pasar utama.")


# ==============================================
# WORKFLOW UTAMA
# ==============================================
if __name__ == "__main__":
    print("\n🚀 Menjalankan pipeline RFM...\n")

    # 1. Cleaning dataset
    output_file = clean_data()
    df_clean = pd.read_csv(output_file, parse_dates=["InvoiceDate"])

    # 2. Eksplorasi awal dataset
    explore_clean_data(df_clean)

    # 3. Feature engineering
    fitur_customer = feature_engineering(df_clean)

    # 4. EDA RFM
    rfm_exploration(fitur_customer, df_clean)

    # 5. Simpan hasil fitur
    fitur_customer.to_csv("feature_customer.csv", index=True)
    print("\n✔ Feature engineering selesai — fitur siap dipakai untuk clustering!")
    print("📁 Disimpan ke: feature_customer.csv")
    print("🧩 Kolom fitur:", list(fitur_customer.columns))
