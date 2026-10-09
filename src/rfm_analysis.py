"""
rfm_analysis.py
===============
Modular utilities for Customer-Level Feature Engineering, RFM Analysis,
Quantile Scoring, and Log/Scaler Preprocessing for Clustering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def calculate_rfm(df: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Calculate Customer-Level Recency, Frequency, and Monetary (RFM) metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing identified customer transactions.
    reference_date : pd.Timestamp, optional
        Reference cutoff date. If None, max(InvoiceDate) + 1 day is used.

    Returns
    -------
    pd.DataFrame
        Customer-level RFM dataframe.
    """
    df_clean = df.copy()
    if 'TotalLineRevenue' not in df_clean.columns:
        df_clean['TotalLineRevenue'] = df_clean['Quantity'] * df_clean['UnitPrice']

    if reference_date is None:
        reference_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df_clean.groupby('CustomerID').agg(
        LastInvoice=('InvoiceDate', 'max'),
        FirstInvoice=('InvoiceDate', 'min'),
        TotalInvoices=('InvoiceNo', 'nunique'),
        PurchaseInvoices=('InvoiceNo', lambda x: x[~x.astype(str).str.startswith('C')].nunique()),
        CancellationInvoices=('InvoiceNo', lambda x: x[x.astype(str).str.startswith('C')].nunique()),
        Monetary=('TotalLineRevenue', 'sum'),
        TotalQuantity=('Quantity', 'sum')
    ).reset_index()

    rfm['Recency'] = (reference_date - rfm['LastInvoice']).dt.days
    rfm['Frequency'] = rfm['PurchaseInvoices'].clip(lower=1)
    rfm['TenureDays'] = (reference_date - rfm['FirstInvoice']).dt.days
    rfm['AvgOrderValue'] = (rfm['Monetary'] / rfm['Frequency']).round(2)

    return rfm


def score_rfm(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign 1-5 quantile scores to Recency, Frequency, and Monetary metrics.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Customer-level RFM dataframe.

    Returns
    -------
    pd.DataFrame
        RFM dataframe with R_Score, F_Score, M_Score, RFM_Segment, and RFM_Score.
    """
    rfm = rfm_df.copy()

    # Recency: lower days -> higher score (5 = most recent)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)

    # Frequency: higher count -> higher score (use rank method='first' to handle discrete ties)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Monetary: higher revenue -> higher score (use rank method='first' to handle tie values)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    return rfm


def preprocess_rfm_for_clustering(rfm_df: pd.DataFrame, filter_positive_monetary: bool = True):
    """
    Apply Log-Transformation (np.log1p) and StandardScaler to RFM features.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Scored or raw RFM dataframe.
    filter_positive_monetary : bool
        If True, filter out customers with Monetary <= 0.

    Returns
    -------
    tuple (pd.DataFrame, np.ndarray, StandardScaler)
        - Processed RFM DataFrame with log columns
        - Scaled feature matrix X_scaled
        - Fitted StandardScaler instance
    """
    rfm = rfm_df.copy()
    if filter_positive_monetary:
        rfm = rfm[rfm['Monetary'] > 0].copy()

    rfm['R_log'] = np.log1p(rfm['Recency'])
    rfm['F_log'] = np.log1p(rfm['Frequency'])
    rfm['M_log'] = np.log1p(rfm['Monetary'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[['R_log', 'F_log', 'M_log']])

    return rfm, X_scaled, scaler
