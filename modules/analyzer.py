import numpy as np
import pandas as pd

def dataset_summary(df):
    numeric = df.select_dtypes(include=np.number).columns
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns
    return {"rows": len(df), "columns": len(df.columns), "numeric_columns": len(numeric), "categorical_columns": len(categorical)}

def numeric_summary(df):
    numeric = df.select_dtypes(include=np.number)
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.reset_index().rename(columns={"index": "Column"})

def correlation_matrix(df):
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr()

def detect_outliers(df):
    results = []
    for col in df.select_dtypes(include=np.number).columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(.25), series.quantile(.75)
        iqr = q3 - q1
        count = 0 if iqr == 0 else int(((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum())
        results.append({"Column": col, "Outliers": count})
    return results
