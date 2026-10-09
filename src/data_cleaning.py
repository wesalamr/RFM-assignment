"""
data_cleaning.py
================
Data cleaning and preprocessing module for Online Retail dataset.
Provides modular functions for loading, cleaning, and filtering retail transaction data.
"""

import pandas as pd
import numpy as np


def load_cleaned_data(filepath: str = "data/interim_structural.parquet") -> pd.DataFrame:
    """
    Load the structurally cleaned online retail dataset.

    Parameters
    ----------
    filepath : str
        Path to the parquet or excel file.

    Returns
    -------
    pd.DataFrame
        Cleaned transaction dataframe with engineered temporal features.
    """
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format for {filepath}")

    # Ensure types
    if 'InvoiceDate' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    if 'Description' in df.columns:
        df['Description'] = df['Description'].astype('string').str.strip()

    return df


def get_identified_customer_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to include only transactions with valid, non-null CustomerIDs.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe containing identified customer transactions with integer CustomerIDs.
    """
    df_known = df[df['CustomerID'].notna()].copy()
    df_known['CustomerID'] = df_known['CustomerID'].astype(int)

    # Ensure TotalLineRevenue calculation
    if 'TotalLineRevenue' not in df_known.columns:
        df_known['TotalLineRevenue'] = df_known['Quantity'] * df_known['UnitPrice']

    return df_known
