import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pointbiserialr, pearsonr
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Fairness Analyzer", layout="wide")

st.title("Fairness Analyzer")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head(20))

    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    st.write("")
    demographic_col = st.selectbox("Select demographic attribute:", df.columns)
    target_col = st.selectbox("Select target variable:", df.columns)

    if demographic_col and target_col:
        groups = df[demographic_col].unique()
        group_means = df.groupby(demographic_col)[target_col].mean()
        ref_group = groups[0]

        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
            rcorr = r - ((r - 1)**2) / (n - 1)
            kcorr = k - ((k - 1)**2) / (n - 1)
            return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

        fairness_results = []
        for grp in groups[1:]:
            spd = group_means[grp] - group_means[ref_group]
            di = (group_means[grp] / group_means[ref_group]) if group_means[ref_group] != 0 else np.nan

            if df[demographic_col].nunique() <= 2 and df[target_col].nunique() <= 2:
                corr, _ = pointbiserialr(df[demographic_col], df[target_col])
            elif df[demographic_col].dtype == 'int64' or df[demographic_col].dtype == 'float64':
                corr, _ = pearsonr(df[demographic_col], df[target_col])
            else:
                corr = cramers_v(df[demographic_col], df[target_col])

            fairness_results.append({
                'Group': grp,
                'Statistical Parity Difference': float(spd),
                'Disparate Impact': float(di) if pd.notna(di) else np.nan,
                'Correlation': float(corr) if pd.notna(corr) else np.nan
            })

        metrics_df = pd.DataFrame(fairness_results)
        metrics_df['Weighted Fairness Score'] = (
            abs(metrics_df['Statistical Parity Difference']) * (1 - abs(metrics_df['Correlation']))
        )

        st.subheader("Fairness Metrics by Group")
        st.dataframe(metrics_df.style.format({
            'Statistical Parity Difference': "{:.3f}",
            'Disparate Impact': "{:.3f}",
            'Correlation': "{:.3f}",
            'Weighted Fairness Score': "{:.3f}"
        }))
