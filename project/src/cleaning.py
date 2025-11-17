# Data Cleaning Module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

COLUMN_ALIASES = {
    'CustomerID': ['CustomerID', 'Customer ID'],
    'UnitPrice': ['UnitPrice', 'Price'],
}


def standardize_columns(df):
    """
    Rename known variant columns to a consistent schema so
    the cleaning pipeline can work across different sources.
    """
    rename_map = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = target
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def clean_data(df):
    """
    Perform data cleaning on the customer segmentation dataset.

    Identifies and addresses:
    - Missing values
    - Duplicate rows
    - Outliers
    - Negative values in quantity/price
    - Date format errors

    Returns:
    - cleaned_df: Cleaned DataFrame
    - summary: Dict with before/after stats
    """
    # Make a copy to avoid modifying original and normalize column names
    df = standardize_columns(df.copy())

    # Initial stats
    initial_shape = df.shape
    initial_missing = df.isnull().sum().sum()
    initial_duplicates = df.duplicated().sum()

    # Identify issues
    price_col = 'UnitPrice' if 'UnitPrice' in df.columns else None
    issues = {
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': initial_duplicates,
        'negative_quantity': (df['Quantity'] < 0).sum() if 'Quantity' in df.columns else 0,
        'negative_unitprice': (df[price_col] < 0).sum() if price_col else 0,
        'outliers_quantity': detect_outliers(df, 'Quantity') if 'Quantity' in df.columns else 0,
        'outliers_unitprice': detect_outliers(df, price_col) if price_col else 0,
        'date_format_errors': check_date_formats(df) if 'InvoiceDate' in df.columns else 0
    }

    # Cleaning process
    # 1. Drop duplicates
    df = df.drop_duplicates()

    # 2. Handle missing values
    # Drop rows with missing CustomerID (critical for segmentation)
    if 'CustomerID' in df.columns:
        # Convert first then drop to avoid object dtype issues
        df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
        df = df.dropna(subset=['CustomerID'])

    # Fill missing Description with 'Unknown'
    if 'Description' in df.columns:
        df['Description'] = df['Description'].fillna('Unknown')

    # Fill missing UnitPrice/Price with median
    if price_col:
        df[price_col] = df[price_col].fillna(df[price_col].median())

    # 3. Filter negative values (remove returns/cancellations if not needed)
    if 'Quantity' in df.columns:
        df = df[df['Quantity'] > 0]
    if price_col:
        df = df[df[price_col] > 0]

    # 4. Convert data types
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        # Drop rows with invalid dates
        df = df.dropna(subset=['InvoiceDate'])

    if 'Quantity' in df.columns:
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    if price_col:
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    if 'CustomerID' in df.columns:
        df['CustomerID'] = df['CustomerID'].astype(int)

    # 5. Handle outliers (cap at 99th percentile)
    if 'Quantity' in df.columns:
        q99 = df['Quantity'].quantile(0.99)
        df['Quantity'] = np.where(df['Quantity'] > q99, q99, df['Quantity'])
    if price_col:
        q99 = df[price_col].quantile(0.99)
        df[price_col] = np.where(df[price_col] > q99, q99, df[price_col])

    # Cap/imputation can create new duplicates, so enforce one more drop
    df = df.drop_duplicates()

    # Final stats
    final_shape = df.shape
    final_missing = df.isnull().sum().sum()
    final_duplicates = df.duplicated().sum()

    summary = {
        'initial_shape': initial_shape,
        'final_shape': final_shape,
        'missing_before': initial_missing,
        'missing_after': final_missing,
        'duplicates_before': initial_duplicates,
        'duplicates_after': final_duplicates,
        'issues_identified': issues,
        'rows_removed': initial_shape[0] - final_shape[0],
        'columns': list(df.columns)
    }

    return df, summary

def detect_outliers(df, column):
    """
    Detect outliers using IQR method.
    """
    if column not in df.columns:
        return 0
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    return outliers

def check_date_formats(df):
    """
    Check for invalid date formats in InvoiceDate.
    """
    if 'InvoiceDate' not in df.columns:
        return 0
    try:
        pd.to_datetime(df['InvoiceDate'], errors='coerce')
        invalid_dates = pd.to_datetime(df['InvoiceDate'], errors='coerce').isnull().sum()
        return invalid_dates
    except:
        return len(df)

def plot_missing_bar(df, title='Missing Values by Column'):
    """
    Plot bar chart of missing values.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        plt.figure(figsize=(10, 6))
        missing.plot(kind='bar')
        plt.title(title)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No missing values found.")

def plot_outliers_boxplot(df, column, title=None):
    """
    Plot boxplot for outliers in a column.
    """
    if column in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[column])
        plt.title(title or f'Boxplot of {column}')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Column '{column}' not found in DataFrame.")


def plot_summary_visuals(raw_df, cleaned_df, column):
    """
    Render bar chart of missing values and boxplots
    (before vs after) in one figure so all appear together.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Missing value bar chart
    missing = raw_df.isnull().sum()
    missing = missing[missing > 0]
    ax = axes[0]
    if len(missing) > 0:
        missing.plot(kind='bar', ax=ax, color=sns.color_palette("muted"))
        ax.set_title('Bar Chart Missing Value (Before Cleaning)')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Tidak ada missing value', ha='center', va='center', fontsize=12)

    # Boxplot before cleaning
    if column in raw_df.columns:
        sns.boxplot(x=raw_df[column], ax=axes[1], color='#FF9551')
        axes[1].set_title(f'Boxplot {column} Sebelum Cleaning')
    else:
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, f'Kolom {column} tidak ada', ha='center', va='center')

    # Boxplot after cleaning
    if column in cleaned_df.columns:
        sns.boxplot(x=cleaned_df[column], ax=axes[2], color='#4E9F3D')
        axes[2].set_title(f'Boxplot {column} Setelah Cleaning')
    else:
        axes[2].axis('off')
        axes[2].text(0.5, 0.5, f'Kolom {column} tidak ada', ha='center', va='center')

    plt.tight_layout()
    plt.show()


def print_issue_summary(summary):
    print("\n✔️ Ringkasan Masalah Data Awal")
    issues = summary['issues_identified']
    print(f"- Total missing value: {int(summary['missing_before']):,}")
    print(f"- Duplikat: {issues['duplicates']:,}")
    if 'Quantity' in summary['columns']:
        print(f"- Quantity negatif: {issues['negative_quantity']:,}")
        print(f"- Outlier Quantity (IQR): {issues['outliers_quantity']:,}")
    price_issue = issues.get('negative_unitprice', 0)
    if price_issue:
        print(f"- Price/UnitPrice negatif: {price_issue:,}")
    out_price = issues.get('outliers_unitprice', 0)
    if out_price:
        print(f"- Outlier Price/UnitPrice (IQR): {out_price:,}")
    if issues.get('date_format_errors', 0):
        print(f"- InvoiceDate invalid: {issues['date_format_errors']:,}")


def print_cleaning_steps():
    print("\n✔️ Penjelasan Langkah-Langkah Cleaning")
    steps = [
        "Drop duplikat agar satu transaksi hanya dihitung sekali.",
        "Konversi dan drop CustomerID invalid lalu isi nilai hilang yang penting.",
        "Filter nilai Quantity/Price negatif yang biasanya retur/cancel.",
        "Konversi tipe data (tanggal & numerik) dan buang tanggal invalid.",
        "Cap outlier di quantile 99 agar statistik tidak ter-distorsi."
    ]
    for idx, step in enumerate(steps, start=1):
        print(f"{idx}. {step}")


def print_before_after(summary):
    print("\n✔️ Perbandingan Sebelum vs Sesudah")
    initial_rows, initial_cols = summary['initial_shape']
    final_rows, final_cols = summary['final_shape']
    print(f"- Bentuk awal: {initial_rows:,} baris x {initial_cols} kolom")
    print(f"- Bentuk akhir: {final_rows:,} baris x {final_cols} kolom")
    print(f"- Missing value: {int(summary['missing_before']):,} -> {int(summary['missing_after']):,}")
    print(f"- Duplikat: {summary['duplicates_before']:,} -> {summary['duplicates_after']:,}")
    print(f"- Total baris dibersihkan: {summary['rows_removed']:,}")


def main():
    data_path = Path(__file__).resolve().parents[1] / "data" / "dataset.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path}")

    raw_df = pd.read_csv(data_path)
    raw_df = standardize_columns(raw_df)
    cleaned_df, summary = clean_data(raw_df)

    print_issue_summary(summary)
    print_cleaning_steps()
    print_before_after(summary)

    target_column = 'Quantity' if 'Quantity' in raw_df.columns else summary['columns'][0]
    print("\nMenyiapkan visualisasi ...")
    plot_summary_visuals(raw_df, cleaned_df, target_column)


if __name__ == "__main__":
    main()
