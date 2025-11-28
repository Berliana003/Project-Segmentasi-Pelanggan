import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import dari file sebelumnya
from normalize_feature import normalize_features
from clean_data import clean_data
from exploration import explore_clean_data
from feature_rfm import feature_engineering
from eda_feature_engineering import eda_feature_engineering

# =====================================================
# METODE ELBOW + SILHOUETTE
# =====================================================
def determine_optimal_clusters(fitur_normalized):
    print("\n===== MENENTUKAN JUMLAH CLUSTER (Elbow & Silhouette) =====\n")

    inertia_values = []     # Untuk Elbow Method
    silhouette_values = []  # Untuk Silhouette Score
    K_range = range(2, 9)   # Coba cluster dari 2 hingga 8 (umumnya 3–5 ideal)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(fitur_normalized)
        inertia_values.append(kmeans.inertia_)

        # Hitung silhouette hanya jika cluster >=2
        score = silhouette_score(fitur_normalized, kmeans.labels_)
        silhouette_values.append(score)

        print(f"K = {k} → Inertia = {kmeans.inertia_:.2f}, Silhouette = {score:.4f}")

    # ==== 1. Plot Elbow Method ====
    plt.figure(figsize=(6, 4))
    plt.plot(K_range, inertia_values, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Jumlah Cluster (K)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()

    # ==== 2. Plot Silhouette ====
    plt.figure(figsize=(6, 4))
    plt.plot(K_range, silhouette_values, marker='o', color='orange')
    plt.title("Silhouette Score")
    plt.xlabel("Jumlah Cluster (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()

    optimal_k = K_range[silhouette_values.index(max(silhouette_values))]
    print(f"\n🎯 Jumlah cluster optimal berdasarkan Silhouette Score = **{optimal_k}**")

    return optimal_k

# =====================================================
# MAIN WORKFLOW CLUSTERING
# =====================================================
if __name__ == "__main__":
    print("\n🚀 Menjalankan CLUSTERING ANALYSIS...\n")

    # 1. Cleaning data
    output_file = clean_data()
    df_clean = pd.read_csv(output_file, parse_dates=["InvoiceDate"])

    # 2. Eksplorasi awal
    explore_clean_data(df_clean)

    # 3. Feature engineering
    fitur_customer = feature_engineering(df_clean)

    # Sesuaikan nama kolom (jika perlu)
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

    print("\n📌 Saran: gunakan jumlah cluster =", optimal_k, "untuk tahap K-Means berikutnya.")
    print("\n🧠 Selanjutnya buat model clustering K-Means final dengan jumlah cluster tersebut.")
