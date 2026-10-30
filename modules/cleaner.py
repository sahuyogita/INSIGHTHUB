def cleaning_report(df):
    return {
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns_with_missing": int((df.isna().sum() > 0).sum()),
    }

def clean_dataset(df, missing_strategy="keep", remove_duplicates=True):
    cleaned = df.copy()
    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()
    if missing_strategy == "drop":
        cleaned = cleaned.dropna()
    elif missing_strategy == "impute":
        for col in cleaned.select_dtypes(include="number").columns:
            if cleaned[col].isna().any():
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        for col in cleaned.select_dtypes(include=["object", "category", "bool"]).columns:
            if cleaned[col].isna().any():
                mode = cleaned[col].mode(dropna=True)
                if not mode.empty:
                    cleaned[col] = cleaned[col].fillna(mode.iloc[0])
    return cleaned
