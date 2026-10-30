import numpy as np
import pandas as pd

def generate_insights(df):
    insights = []
    missing = int(df.isna().sum().sum())
    total = df.shape[0] * df.shape[1]
    if total:
        pct = missing / total * 100
        insights.append(f"{missing:,} missing cells were detected ({pct:.1f}% of all cells)." if missing else "No missing values were detected.")

    duplicates = int(df.duplicated().sum())
    insights.append(f"{duplicates:,} duplicate rows were detected." if duplicates else "No duplicate rows were detected.")

    numeric = df.select_dtypes(include=np.number)
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 3:
            continue
        mean, median = s.mean(), s.median()
        if mean != 0:
            diff = abs(mean - median) / abs(mean) * 100
            if diff >= 20:
                direction = "higher" if mean > median else "lower"
                insights.append(f"{col} may be skewed: its mean is about {diff:.1f}% {direction} than its median.")
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        if iqr:
            count = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
            if count:
                insights.append(f"{col} contains approximately {count:,} IQR-based outliers.")

    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                value = corr.iloc[i, j]
                if pd.notna(value):
                    pairs.append((abs(value), value, cols[i], cols[j]))
        pairs.sort(reverse=True)
        if pairs and pairs[0][0] >= .7:
            _, value, a, b = pairs[0]
            direction = "positive" if value > 0 else "negative"
            insights.append(f"{a} and {b} have a strong {direction} correlation (r = {value:.2f}).")

    categorical = df.select_dtypes(include=["object", "category", "bool"])
    for col in categorical.columns:
        counts = df[col].value_counts(dropna=True)
        if not counts.empty:
            share = counts.iloc[0] / counts.sum() * 100
            if share >= 50:
                insights.append(f"{col}: '{counts.index[0]}' is the dominant category ({share:.1f}% of non-missing values).")
    return insights[:12]
