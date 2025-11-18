import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# import fungsi cleaning
from cleaning import (
    standardize_columns, clean_data,
    print_issue_summary, print_cleaning_steps,
    print_before_after, plot_summary_visuals
)

# ===========================
#   E X P L O R A T I O N
# ===========================

def descriptive_statistics(df):
    print("\n📊 Statistik Deskriptif:")
    print(df.describe(include='all'))

    print("\n📌 Mode:")
    print(df.mode().iloc[0])


def show_correlation(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("\nTidak ada variabel numerik untuk korelasi.")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Heatmap Korelasi")
    plt.show()


def plot_histograms(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("\nTidak ada kolom numerik untuk histogram.")
        return

    df[numeric_cols].hist(figsize=(12, 8), bins=30)
    plt.suptitle("Histogram Variabel Numerik")
    plt.show()


def scatter_example(df):
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df['Quantity'], df['UnitPrice'], alpha=0.4)
        plt.xlabel("Quantity")
        plt.ylabel("UnitPrice")
        plt.title("Scatter: Quantity vs UnitPrice")
        plt.show()


def boxplot_all(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("\nTidak ada kolom numerik untuk boxplot.")
        return

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[numeric_cols])
    plt.title("Boxplot Variabel Numerik")
    plt.xticks(rotation=45)
    plt.show()


def initial_insights(df):
    print("\n📝 Insight Awal:")

    if 'CustomerID' in df.columns:
        print(f"- Jumlah pelanggan unik: {df['CustomerID'].nunique()}")

    if 'Quantity' in df.columns:
        print(f"- Rata-rata Quantity: {df['Quantity'].mean():.2f}")

    if 'UnitPrice' in df.columns:
        print(f"- Rata-rata UnitPrice: {df['UnitPrice'].mean():.2f}")

    if 'Country' in df.columns:
        print(f"- Negara terbanyak transaksi: {df['Country'].mode()[0]}")


def advanced_exploration(cleaned_df):
    # Create TotalSales if not present
    if 'TotalSales' not in cleaned_df.columns:
        cleaned_df['TotalSales'] = cleaned_df['Quantity'] * cleaned_df['UnitPrice']

    numeric_cols = ['Quantity', 'UnitPrice', 'TotalSales']
    stats = cleaned_df[numeric_cols].describe().T
    stats['median'] = cleaned_df[numeric_cols].median()
    stats['mode'] = cleaned_df[numeric_cols].mode().iloc[0]

    agg_map = {
        'total_sales': ('TotalSales', 'sum'),
        'total_items': ('Quantity', 'sum')
    }
    if 'Invoice' in cleaned_df.columns:
        agg_map['invoices'] = ('Invoice', 'nunique')

    customer_summary = cleaned_df.groupby('CustomerID').agg(**agg_map).sort_values('total_sales', ascending=False)

    print('Statistik deskriptif fitur numerik:')
    print(stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'median', 'mode']])

    print('\nTop 5 pelanggan berdasarkan nilai belanja:')
    print(customer_summary.head())

    sample_n = min(10000, len(cleaned_df))
    sample_df = cleaned_df.sample(sample_n, random_state=42) if len(cleaned_df) > sample_n else cleaned_df

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(sample_df['UnitPrice'], sample_df['Quantity'],
                          c=sample_df['TotalSales'], cmap='viridis', alpha=0.4, edgecolors='none')
    plt.colorbar(scatter, label='TotalSales')
    plt.xlabel('UnitPrice')
    plt.ylabel('Quantity')
    plt.title('Scatterplot UnitPrice vs Quantity (warna = TotalSales)')
    plt.show()

    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(18, 5))
    for ax, col in zip(axes, numeric_cols):
        sns.histplot(cleaned_df[col], bins=40, ax=ax, kde=True, color='#1f77b4')
        ax.set_title(f'Histogram {col}')
    plt.tight_layout()
    plt.show()

    top_countries = cleaned_df.groupby('Country')['TotalSales'].sum().sort_values(ascending=False).head(5).index
    subset = cleaned_df[cleaned_df['Country'].isin(top_countries)]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=subset, x='Country', y='TotalSales')
    plt.title('Boxplot TotalSales untuk 5 Negara Teratas')
    plt.ylabel('TotalSales per transaksi')
    plt.xlabel('Country')
    plt.xticks(rotation=45)
    plt.show()


# ===========================
#           MAIN
# ===========================

def main():
    data_path = Path(__file__).resolve().parents[1] / "data" / "dataset.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path}")

    raw_df = pd.read_csv(data_path)
    raw_df = standardize_columns(raw_df)

    print("\n========================")
    print("📌 DATA EXPLORATION")
    print("========================")

    descriptive_statistics(raw_df)
    show_correlation(raw_df)
    plot_histograms(raw_df)
    scatter_example(raw_df)
    boxplot_all(raw_df)
    initial_insights(raw_df)

    print("\n========================")
    print("📌 DATA CLEANING")
    print("========================")

    cleaned_df, summary = clean_data(raw_df)

    print_issue_summary(summary)
    print_cleaning_steps()
    print_before_after(summary)

    advanced_exploration(cleaned_df)

    target_column = 'Quantity' if 'Quantity' in raw_df.columns else summary['columns'][0]
    plot_summary_visuals(raw_df, cleaned_df, target_column)


if __name__ == "__main__":
    main()
