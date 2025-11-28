import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist  # untuk hitung jarak centroid

# IMPORT dari file sebelumnya
from clean_data import clean_data
from exploration import explore_clean_data
from feature_rfm import feature_engineering
from eda_feature_engineering import eda_feature_engineering
from normalize_feature import normalize_features
from clustering import determine_optimal_clusters

# =====================================================
# FINAL CLUSTERING K-MEANS
# =====================================================
def final_kmeans_clustering(fitur_normalized, optimal_k, fitur_customer, df_clean):
    print(f"\n===== 🚀 MEMBUAT MODEL K-MEANS DENGAN K = {optimal_k} =====\n")

    # 1. Buat model dan fit
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(fitur_normalized)

    # 2. Tambahkan ke dataset
    fitur_customer["Cluster"] = cluster_labels
    df_clean["Cluster"] = df_clean["Customer_ID"].map(fitur_customer["Cluster"])

    print("\n📌 Contoh hasil clustering:")
    print(fitur_customer.head())

    # 3. Analisis per cluster
    cluster_summary = fitur_customer.groupby("Cluster").mean()
    print("\n📊 Ringkasan rata-rata per cluster:")
    print(cluster_summary)

    # 4. Cari negara dominan tiap cluster
    country_cluster = df_clean.groupby("Cluster")["Country"].agg(lambda x: x.mode()[0])
    print("\n🌍 Negara dominan per cluster:")
    print(country_cluster)

    # 5. Interpretasi cluster
    print("\n🔍 INTERPRETASI CLUSTER:")
    for cluster in cluster_summary.index:
        rec = cluster_summary.loc[cluster, "Recency_Days"]
        freq = cluster_summary.loc[cluster, "Total_Transactions"]
        mon = cluster_summary.loc[cluster, "Total_Spending"]
        country = country_cluster[cluster]

        interpretasi = []
        if rec < fitur_customer["Recency_Days"].mean():
            interpretasi.append("➡ Aktif (Recency rendah)")
        else:
            interpretasi.append("➡ Tidak aktif (Recency tinggi)")

        if freq > fitur_customer["Total_Transactions"].mean():
            interpretasi.append("➡ Sering belanja (Frequency tinggi)")
        else:
            interpretasi.append("➡ Jarang belanja (Frequency rendah)")

        if mon > fitur_customer["Total_Spending"].mean():
            interpretasi.append("➡ Big spender (Monetary tinggi)")

        print(f"\n🟦 Cluster {cluster}:")
        print(" | ".join(interpretasi))
        print(f"🌍 Negara dominan: {country}")

    # =====================================================
    # Tambahan: Hitung jarak antar centroid
    # =====================================================
    centroids = kmeans.cluster_centers_
    centroid_distances = cdist(centroids, centroids)

    print("\n📏 MATRIX JARAK ANTAR CENTROID (semakin besar → cluster makin terpisah):")
    print(pd.DataFrame(centroid_distances, 
                       index=[f"Cluster {i}" for i in range(optimal_k)],
                       columns=[f"Cluster {i}" for i in range(optimal_k)]))

    print("\n📎 Koordinat centroid (dalam fitur yang sudah dinormalisasi):")
    for idx, center in enumerate(centroids):
        print(f"Cluster {idx} → {center}")

    # =====================================================
    # Visualisasi Clustering + Centroid
    # =====================================================
    plt.figure(figsize=(7, 5))
    scatter = sns.scatterplot(
        x=fitur_normalized["Recency_Days"],
        y=fitur_normalized["Total_Spending"],
        hue=cluster_labels,
        palette="viridis",
        style=cluster_labels
    )

    # Tambahkan titik pusat centroid
    plt.scatter(
        centroids[:, fitur_normalized.columns.get_loc("Recency_Days")],
        centroids[:, fitur_normalized.columns.get_loc("Total_Spending")],
        s=200, c='red', marker='X', label='Centroid'
    )

    plt.title("Visualisasi Clustering (Recency vs Monetary) + Centroid")
    plt.xlabel("Recency (Normalized)")
    plt.ylabel("Monetary (Normalized)")
    plt.legend()
    plt.show()

    return fitur_customer, kmeans


# =====================================================
# MAIN RUN
# =====================================================
if __name__ == "__main__":
    print("\n🚀 FINAL CLUSTERING ANALYSIS...\n")

    # 1. Cleaning data
    output_file = clean_data()
    df_clean = pd.read_csv(output_file, parse_dates=["InvoiceDate"])

    # 2. Explorasi data
    explore_clean_data(df_clean)

    # 3. Feature engineering
    fitur_customer = feature_engineering(df_clean)
    fitur_customer = fitur_customer.rename(columns={
        "Recency": "Recency_Days",
        "Frequency": "Total_Transactions",
        "Monetary": "Total_Spending"
    })

    # 4. EDA Feature Engineering
    eda_feature_engineering(fitur_customer, df_clean)

    # 5. Normalisasi
    fitur_normalized, scaler = normalize_features(fitur_customer)

    # 6. Tentukan jumlah cluster
    optimal_k = determine_optimal_clusters(fitur_normalized)

    # 7. Clustering final
    fitur_hasil_cluster, model_kmeans = final_kmeans_clustering(
        fitur_normalized, optimal_k, fitur_customer, df_clean
    )

    # 8. Simpan hasil
    fitur_hasil_cluster.to_csv("customer_cluster_result.csv", index=True)
    print("\n📁 Hasil clustering disimpan ke: customer_cluster_result.csv")
