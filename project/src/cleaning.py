# Data Cleaning Module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    # Make a copy to avoid modifying original
    df = df.copy()

    # Initial stats
    initial_shape = df.shape
    initial_missing = df.isnull().sum().sum()
    initial_duplicates = df.duplicated().sum()

    # Identify issues
    issues = {
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': initial_duplicates,
        'negative_quantity': (df['Quantity'] < 0).sum() if 'Quantity' in df.columns else 0,
        'negative_unitprice': (df['UnitPrice'] < 0).sum() if 'UnitPrice' in df.columns else 0,
        'outliers_quantity': detect_outliers(df, 'Quantity') if 'Quantity' in df.columns else 0,
        'outliers_unitprice': detect_outliers(df, 'UnitPrice') if 'UnitPrice' in df.columns else 0,
        'date_format_errors': check_date_formats(df) if 'InvoiceDate' in df.columns else 0
    }

    # Cleaning process
    # 1. Drop duplicates
    df = df.drop_duplicates()

    # 2. Handle missing values
    # Drop rows with missing CustomerID (critical for segmentation)
    if 'CustomerID' in df.columns:
        df = df.dropna(subset=['CustomerID'])

    # Fill missing Description with 'Unknown'
    if 'Description' in df.columns:
        df['Description'] = df['Description'].fillna('Unknown')

    # Fill missing UnitPrice with median
    if 'UnitPrice' in df.columns:
        df['UnitPrice'] = df['UnitPrice'].fillna(df['UnitPrice'].median())

    # 3. Filter negative values (remove returns/cancellations if not needed)
    if 'Quantity' in df.columns:
        df = df[df['Quantity'] > 0]
    if 'UnitPrice' in df.columns:
        df = df[df['UnitPrice'] > 0]

    # 4. Convert data types
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        # Drop rows with invalid dates
        df = df.dropna(subset=['InvoiceDate'])

    if 'Quantity' in df.columns:
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    if 'UnitPrice' in df.columns:
        df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    if 'CustomerID' in df.columns:
        df['CustomerID'] = df['CustomerID'].astype(int, errors='ignore')

    # 5. Handle outliers (cap at 99th percentile)
    if 'Quantity' in df.columns:
        q99 = df['Quantity'].quantile(0.99)
        df['Quantity'] = np.where(df['Quantity'] > q99, q99, df['Quantity'])
    if 'UnitPrice' in df.columns:
        q99 = df['UnitPrice'].quantile(0.99)
        df['UnitPrice'] = np.where(df['UnitPrice'] > q99, q99, df['UnitPrice'])

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

def plot_missing_bar(df):
    """
    Plot bar chart of missing values.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        plt.figure(figsize=(10, 6))
        missing.plot(kind='bar')
        plt.title('Missing Values by Column')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("No missing values found.")

def plot_outliers_boxplot(df, column):
    """
    Plot boxplot for outliers in a column.
    """
    if column in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column}')
        plt.show()
    else:
        print(f"Column '{column}' not found in DataFrame.")
