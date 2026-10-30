import streamlit as st
import pandas as pd
import numpy as np

from modules.loader import load_dataset
from modules.cleaner import cleaning_report, clean_dataset
from modules.analyzer import dataset_summary, numeric_summary, correlation_matrix, detect_outliers
from modules.visualizer import plot_histogram, plot_boxplot, plot_countplot, plot_scatter, plot_correlation_heatmap
from modules.insights import generate_insights

st.set_page_config(page_title="INSIGHTHUB", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.brand {font-size: 2.7rem; font-weight: 800; letter-spacing: -1px;}
.tagline {font-size: 1rem; color: #707070; margin-bottom: 1.5rem;}
.insight {border-left: 4px solid #6c63ff; padding: 12px 16px; margin: 9px 0; border-radius: 8px; background: rgba(108,99,255,.07);}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="brand">📊 INSIGHTHUB</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Turn raw datasets into clean analysis, visualizations, and actionable insights.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📂 Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    st.divider()
    page = st.radio("Navigate", ["🏠 Dashboard", "📋 Data", "🧹 Cleaning", "📈 Statistics", "📊 Visualizations", "🔗 Correlations", "💡 Insights"])
    st.divider()
    st.caption("INSIGHTHUB v2.0")
    st.caption("Pandas • NumPy • Matplotlib • Seaborn • Streamlit")

if uploaded_file is None:
    st.markdown("### 👋 Welcome to INSIGHTHUB")
    st.info("Upload a CSV or Excel dataset from the sidebar to begin.")
    a, b, c = st.columns(3)
    a.info("🧹 **Clean**\n\nDetect missing values, duplicates and outliers.")
    b.info("📊 **Analyze**\n\nCalculate statistics and relationships.")
    c.info("💡 **Understand**\n\nGenerate visualizations and insights.")
    st.stop()

try:
    df = load_dataset(uploaded_file)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if df.empty:
    st.warning("The uploaded dataset is empty.")
    st.stop()

summary = dataset_summary(df)
report = cleaning_report(df)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if not st.session_state.analysis_done:
    st.markdown("### 🔍 Ready to analyze")
    st.write(f"**{uploaded_file.name}** loaded successfully. Click the button to generate the complete analysis.")
    if st.button("🚀 Analyze Dataset", type="primary", use_container_width=True):
        st.session_state.analysis_done = True
        st.rerun()
    st.stop()

if page == "🏠 Dashboard":
    st.subheader("📌 Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{summary['rows']:,}")
    c2.metric("Columns", summary["columns"])
    c3.metric("Missing Cells", f"{report['missing_cells']:,}")
    c4.metric("Duplicates", f"{report['duplicate_rows']:,}")
    c5.metric("Numeric", summary["numeric_columns"])

    st.markdown("### 📋 Quick Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 💡 Quick Insights")
    for insight in generate_insights(df)[:5]:
        st.markdown(f'<div class="insight">💡 {insight}</div>', unsafe_allow_html=True)

elif page == "📋 Data":
    st.subheader("📋 Data Explorer")
    st.dataframe(df, use_container_width=True, height=520)
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing": df.isna().sum().values,
        "Unique": [df[c].nunique(dropna=True) for c in df.columns],
    })
    st.subheader("Column Information")
    st.dataframe(info_df, use_container_width=True)

elif page == "🧹 Cleaning":
    st.subheader("🧹 Data Cleaning")
    a, b, c = st.columns(3)
    a.metric("Missing Cells", f"{report['missing_cells']:,}")
    b.metric("Duplicate Rows", f"{report['duplicate_rows']:,}")
    c.metric("Columns With Missing", report["columns_with_missing"])

    strategy_label = st.selectbox("Missing-value handling", ["Keep unchanged", "Drop rows", "Fill numeric with median + categorical with mode"])
    remove_duplicates = st.checkbox("Remove duplicate rows", value=True)

    if st.button("🧹 Apply Cleaning", type="primary"):
        strategy = {"Keep unchanged": "keep", "Drop rows": "drop", "Fill numeric with median + categorical with mode": "impute"}[strategy_label]
        cleaned = clean_dataset(df, missing_strategy=strategy, remove_duplicates=remove_duplicates)
        st.session_state["cleaned_df"] = cleaned
        st.success(f"Cleaning finished. New shape: {cleaned.shape[0]:,} rows × {cleaned.shape[1]} columns.")

    cleaned = st.session_state.get("cleaned_df")
    if cleaned is not None:
        st.dataframe(cleaned.head(100), use_container_width=True)
        st.download_button("⬇️ Download Cleaned CSV", cleaned.to_csv(index=False).encode("utf-8"), "insighthub_cleaned.csv", "text/csv")

elif page == "📈 Statistics":
    st.subheader("📈 Descriptive Statistics")
    stats = numeric_summary(df)
    if stats.empty:
        st.warning("No numeric columns detected.")
    else:
        st.dataframe(stats, use_container_width=True)
    st.subheader("🚨 Outlier Detection")
    outliers = detect_outliers(df)
    if outliers:
        st.dataframe(pd.DataFrame(outliers), use_container_width=True)
    else:
        st.success("No IQR-based outliers detected.")

elif page == "📊 Visualizations":
    st.subheader("📊 Visualization Studio")
    if numeric_cols:
        selected = st.selectbox("Numeric column", numeric_cols)
        a, b = st.columns(2)
        with a:
            st.pyplot(plot_histogram(df, selected), use_container_width=True)
        with b:
            st.pyplot(plot_boxplot(df, selected), use_container_width=True)
    if categorical_cols:
        selected_cat = st.selectbox("Categorical column", categorical_cols)
        st.pyplot(plot_countplot(df, selected_cat), use_container_width=True)
    if len(numeric_cols) >= 2:
        st.markdown("### 🔵 Relationship Explorer")
        a, b = st.columns(2)
        with a:
            x_col = st.selectbox("X axis", numeric_cols)
        with b:
            y_col = st.selectbox("Y axis", numeric_cols, index=1)
        st.pyplot(plot_scatter(df, x_col, y_col), use_container_width=True)

elif page == "🔗 Correlations":
    st.subheader("🔗 Correlation Analysis")
    corr = correlation_matrix(df)
    if corr.empty:
        st.warning("At least two numeric columns are required.")
    else:
        st.pyplot(plot_correlation_heatmap(corr), use_container_width=True)
        st.dataframe(corr.round(3), use_container_width=True)

elif page == "💡 Insights":
    st.subheader("💡 INSIGHTHUB Insights")
    st.write("Automatically generated observations based on the uploaded dataset.")
    insights = generate_insights(df)
    for i, insight in enumerate(insights, 1):
        st.markdown(f'<div class="insight"><b>Insight {i}</b><br>{insight}</div>', unsafe_allow_html=True)
    st.divider()
    st.caption("Statistical observations are not proof of causation.")
