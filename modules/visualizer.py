import matplotlib.pyplot as plt
import seaborn as sns

def _fig():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.tight_layout()
    return fig, ax

def plot_histogram(df, column):
    fig, ax = _fig()
    sns.histplot(df[column].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    return fig

def plot_boxplot(df, column):
    fig, ax = _fig()
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f"Box Plot of {column}")
    return fig

def plot_countplot(df, column):
    fig, ax = _fig()
    counts = df[column].astype(str).value_counts().head(20)
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title(f"Top Categories in {column}")
    return fig

def plot_scatter(df, x_col, y_col):
    fig, ax = _fig()
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f"{y_col} vs {x_col}")
    return fig

def plot_correlation_heatmap(corr):
    fig, ax = _fig()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig
