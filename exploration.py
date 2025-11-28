import pandas as pd
from clean_data import clean_data

def explore_clean_data(df: pd.DataFrame):
    print("\n===== EXPLORASI DATA BERSIH =====")

    # 1. Distribusi nilai transaksi (TotalPrice = Quantity × Price)
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    print("\n▶ Statistik Distribusi Nilai Transaksi")
    print(df["TotalPrice"].describe())

    # 2. Distribusi jumlah transaksi per invoice
    transaksi_per_invoice = df.groupby("Invoice")["StockCode"].count()
    print("\n▶ Distribusi Jumlah Item per Transaksi (Invoice)")
    print(transaksi_per_invoice.describe())

    # 3. Tren transaksi per bulan
    df["Month"] = df["InvoiceDate"].dt.to_period("M")
    transaksi_per_bulan = df.groupby("Month")["Invoice"].nunique()
    print("\n▶ Jumlah Transaksi per Bulan")
    print(transaksi_per_bulan)

    # 4. Tren transaksi per jam
    df["Hour"] = df["InvoiceDate"].dt.hour
    transaksi_per_jam = df.groupby("Hour")["Invoice"].nunique()
    print("\n▶ Jumlah Transaksi berdasarkan Jam")
    print(transaksi_per_jam)

    # 5. Return
    print("\n▶ Return sudah dibersihkan (Quantity negatif tidak ada).")

    # ==========================
    # Analisis Customer
    # ==========================
    print("\n===== ANALISIS CUSTOMER =====")

    if "Customer_ID" in df.columns:
        transaksi_per_customer = df.groupby("Customer_ID")["Invoice"].nunique()
        print("\n▶ Customer dengan transaksi terbanyak:")
        print(transaksi_per_customer.sort_values(ascending=False).head())

        active_customers = transaksi_per_customer.count()
        one_time_customers = (transaksi_per_customer == 1).sum()

        print(f"\n▶ Total Customer Aktif          : {active_customers}")
        print(f"▶ Customer hanya beli 1 kali    : {one_time_customers}")
    else:
        print("\n⚠ Tidak ada kolom 'Customer_ID' untuk analisis customer.")

    # ==========================
    # Analisis Produk
    # ==========================
    print("\n===== ANALISIS PRODUK =====")

    produk_terjual = df.groupby("StockCode")["Quantity"].sum()
    print("\n▶ Produk paling banyak dijual (berdasarkan Quantity):")
    print(produk_terjual.sort_values(ascending=False).head())

    produk_rata_harga = df.groupby("StockCode")["Price"].mean().sort_values(ascending=False).head()
    print("\n▶ Produk dengan rata-rata harga tertinggi:")
    print(produk_rata_harga)

    print("\n📌 Insight umum produk:")
    print("- Produk dengan total Quantity tinggi → produk populer.")
    print("- Produk yang sering dibeli bisa menjadi target promosi.")
    print("- Rata-rata harga produk mendukung strategi pricing.")
    
    print("\n📌 Insight umum customer:")
    print("- Customer yang sering membeli bisa jadi target loyalitas.")
    print("- Customer yang hanya beli sekali bisa dianalisis alasannya.")
    print("- Segmen pelanggan aktif penting untuk retensi.")

# ==========================================
# Pemanggilan fungsi setelah cleaning
# ==========================================
if __name__ == "__main__":
    output_file = clean_data()
    print(f"Cleaned data saved to: {output_file}")

    df_clean = pd.read_csv(output_file, parse_dates=["InvoiceDate"])
    explore_clean_data(df_clean)
