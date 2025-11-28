import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import dari file sebelumnya
from clean_data import clean_data
from exploration import explore_clean_data
from feature_rfm import feature_engineering
from eda_feature_engineering import eda_feature_engineering  

# =====================================================
# NORMALISASI DATA (StandardScaler)
# =====================================================
def normalize_features(fitur_customer: pd.DataFrame):
    print("\n===== NORMALISASI DATA (StandardScaler) =====\n")

    scaler = StandardScaler()
    fitur_normalized = scaler.fit_transform(fitur_customer)

    fitur_normalized = pd.DataFrame(
        fitur_normalized,
        index=fitur_customer.index,
        columns=fitur_customer.columns
    )

    print("📊 Contoh sebelum normalisasi:")
    print(fitur_customer.head())

    print("\n📈 Contoh setelah normalisasi:")
    print(fitur_normalized.head())

    print("\n✔ Normalisasi selesai — data siap digunakan untuk clustering (K-Means).")
    print("⚠ Pastikan tidak ada outlier ekstrim sebelum clustering.")

    return fitur_normalized, scaler

# =====================================================
# MAIN WORKFLOW 
# =====================================================
if __name__ == "__main__":
    print("\n🚀 Menjalankan FULL PIPELINE (Cleaning → EDA → RFM → Normalisasi)...\n")

    # 1. Cleaning data
    output_file = clean_data()
    df_clean = pd.read_csv(output_file, parse_dates=["InvoiceDate"])

    # 2. Eksplorasi awal dataset
    explore_clean_data(df_clean)

    # 3. Feature Engineering
    fitur_customer = feature_engineering(df_clean)

    # Catatan: pastikan nama kolom sesuai dengan EDA jika kamu rename sebelumnya
    fitur_customer = fitur_customer.rename(columns={
        "Recency": "Recency_Days",
        "Frequency": "Total_Transactions",
        "Monetary": "Total_Spending"
    })

    # 4. EDA Feature Engineering
    eda_feature_engineering(fitur_customer, df_clean)

    # 5. Normalisasi Data
    fitur_normalized, scaler = normalize_features(fitur_customer)

    # 6. Simpan hasil normalisasi
    fitur_normalized.to_csv("feature_normalized.csv")
    print("\n📁 Fitur hasil normalisasi disimpan ke: feature_normalized.csv")

    print("\n🧠 Pipeline selesai — tinggal lanjut ke CLUSTERING (K-Means).")
