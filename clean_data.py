from pathlib import Path
import pandas as pd
import numpy as np

def clean_data(input_path: str = "dataset.csv", output_path: str = "cleaned_dataset.csv") -> Path:
    """
    Load dataset, apply cleaning rules, and perform EDA checks:
    - Missing value analysis (per kolom & total)
    - Duplicate check
    - Negative quantity (returns)
    - Price & Quantity outliers (IQR method)
    - Abnormal invoice date detection
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(
        input_path,
        dtype={"Invoice": str, "StockCode": str},
        low_memory=False,
    )

    # ========== EDA BEFORE CLEANING ==========
    print("===== EDA AWAL (SEBELUM CLEANING) =====")

    # 1. Missing value
    print("\n▶ Missing Value per Kolom:")
    print(df.isnull().sum())
    print(f"\n▶ TOTAL Missing Value: {df.isnull().sum().sum()}")

    # 2. Duplikasi
    print(f"\n▶ Jumlah Duplikasi Baris: {df.duplicated().sum()}")

    # 3. Transaksi dengan Quantity negatif
    if "Quantity" in df.columns:
        print(f"\n▶ Jumlah Quantity Negatif (return): {(df['Quantity'] < 0).sum()}")

    # 4. Outlier Harga (IQR)
    if "Price" in df.columns:
        Q1_p = df["Price"].quantile(0.25)
        Q3_p = df["Price"].quantile(0.75)
        IQR_p = Q3_p - Q1_p
        outliers_price = ((df["Price"] < (Q1_p - 1.5 * IQR_p)) | (df["Price"] > (Q3_p + 1.5 * IQR_p))).sum()
        print(f"\n▶ Jumlah Outlier Harga: {outliers_price}")

    # 5. Outlier Quantity (IQR)
    if "Quantity" in df.columns:
        Q1_q = df["Quantity"].quantile(0.25)
        Q3_q = df["Quantity"].quantile(0.75)
        IQR_q = Q3_q - Q1_q
        outliers_qty = ((df["Quantity"] < (Q1_q - 1.5 * IQR_q)) | (df["Quantity"] > (Q3_q + 1.5 * IQR_q))).sum()
        print(f"\n▶ Jumlah Outlier Quantity (IQR): {outliers_qty}")

    # 6. Invoice Date aneh
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        abnormal_date_count = ((df["InvoiceDate"] < "1900-01-01") | (df["InvoiceDate"] > pd.Timestamp.now())).sum()
        print(f"\n▶ Jumlah Invoice dengan Tanggal Aneh: {abnormal_date_count}")

    # ========== DATA CLEANING ==========
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df = df.dropna(subset=["Invoice", "StockCode", "Quantity", "Price"])
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df = df[~df["Invoice"].str.startswith("C")]

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Quantity", "Price"])
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    if "Description" in df.columns:
        df["Description"] = df["Description"].astype(str).str.strip()
        df = df[df["Description"] != ""]

    if "Customer_ID" in df.columns:
        df = df.dropna(subset=["Customer_ID"])

    dedup_keys = ["Invoice", "StockCode", "InvoiceDate", "Quantity", "Price"]
    existing_keys = [k for k in dedup_keys if k in df.columns]
    df = df.drop_duplicates(subset=existing_keys)

    df = df.sort_values(by=["Invoice", "StockCode", "InvoiceDate"]).reset_index(drop=True)

    # ========== EDA SETELAH CLEANING ==========
    print("\n===== EDA SETELAH CLEANING =====")

    # 1. Missing value
    print("\n▶ Missing Value per Kolom:")
    print(df.isnull().sum())
    print(f"\n▶ TOTAL Missing Value: {df.isnull().sum().sum()}")

    # 2. Duplikasi
    print(f"\n▶ Jumlah Duplikasi Baris: {df.duplicated().sum()}")

    # 3. Transaksi dengan Quantity negatif
    if "Quantity" in df.columns:
        print(f"\n▶ Jumlah Quantity Negatif (return): {(df['Quantity'] < 0).sum()}")

    # 4. Outlier Harga (IQR)
    if "Price" in df.columns:
        Q1_p = df["Price"].quantile(0.25)
        Q3_p = df["Price"].quantile(0.75)
        IQR_p = Q3_p - Q1_p
        outliers_price = ((df["Price"] < (Q1_p - 1.5 * IQR_p)) | (df["Price"] > (Q3_p + 1.5 * IQR_p))).sum()
        print(f"\n▶ Jumlah Outlier Harga: {outliers_price}")

    # 5. Outlier Quantity (IQR)
    if "Quantity" in df.columns:
        Q1_q = df["Quantity"].quantile(0.25)
        Q3_q = df["Quantity"].quantile(0.75)
        IQR_q = Q3_q - Q1_q
        outliers_qty = ((df["Quantity"] < (Q1_q - 1.5 * IQR_q)) | (df["Quantity"] > (Q3_q + 1.5 * IQR_q))).sum()
        print(f"\n▶ Jumlah Outlier Quantity (IQR): {outliers_qty}")

    # 6. Invoice Date aneh
    if "InvoiceDate" in df.columns:
        abnormal_date_count = ((df["InvoiceDate"] < "1900-01-01") | (df["InvoiceDate"] > pd.Timestamp.now())).sum()
        print(f"\n▶ Jumlah Invoice dengan Tanggal Aneh: {abnormal_date_count}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n✔ Cleaning selesai. Data disimpan ke:", output_path)
    return output_path


if __name__ == "__main__":
    output_file = clean_data()
    print(f"Cleaned data saved to: {output_file}")
