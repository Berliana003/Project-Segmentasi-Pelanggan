from pathlib import Path
import pandas as pd


def clean_data(input_path: str = "dataset.csv", output_path: str = "cleaned_dataset.csv") -> Path:
    """Load dataset, apply simple cleaning rules, and write a new CSV.

    Rules:
    - Strip/normalize column names (spaces->underscores).
    - Drop exact duplicate rows.
    - Coerce numeric fields and remove non-positive quantities/prices.
    - Parse dates and drop rows with missing critical fields.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(
        input_path,
        dtype={"Invoice": str, "StockCode": str},  # keep IDs as strings to avoid mixed-type warnings
        low_memory=False,
    )

    # Normalize column names to make downstream processing easier.
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Remove rows missing critical identifiers or amounts.
    df = df.dropna(subset=["Invoice", "StockCode", "Quantity", "Price"])

    # Keep identifiers as trimmed strings for consistency.
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()

    # Drop obvious cancellations where invoice starts with "C".
    df = df[~df["Invoice"].str.startswith("C")]

    # Coerce numeric columns; invalid parsing becomes NaN and is removed.
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Quantity", "Price"])

    # Remove rows with non-positive quantity or price (likely cancellations/bad entries).
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    # Parse invoice date and drop rows where parsing fails.
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # Trim description whitespace and drop empty descriptions.
    if "Description" in df.columns:
        df["Description"] = df["Description"].astype(str).str.strip()
        df = df[df["Description"] != ""]

    # Drop rows without a customer id if present.
    if "Customer_ID" in df.columns:
        df = df.dropna(subset=["Customer_ID"])

    # Drop duplicate rows based on key fields.
    dedup_keys = ["Invoice", "StockCode", "InvoiceDate", "Quantity", "Price"]
    existing_keys = [k for k in dedup_keys if k in df.columns]
    df = df.drop_duplicates(subset=existing_keys)

    # Sort for reproducibility.
    df = df.sort_values(by=["Invoice", "StockCode", "InvoiceDate"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    output_file = clean_data()
    print(f"Cleaned data saved to: {output_file}")
