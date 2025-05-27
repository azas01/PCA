import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from kneed import KneeLocator

st.title("PCA Dimensionality Reduction App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Chỉ lấy các cột số
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found in the dataset after cleaning.")
        st.stop()

    st.subheader("Settings")
    select_cols = st.multiselect(
        label="Select numeric features for PCA", 
        options=numeric_cols,
        default=numeric_cols
    )
    method = st.radio(
        label="Component selection method",
        options=["Elbow Method", "Kaiser Criterion"],
        index=0
    )

    if select_cols:
        data_numeric = df[select_cols]
        # Chuẩn hóa dữ liệu 
        scaled_data = preprocessing.scale(data_numeric)
        st.write(f"**Data shape before PCA:** {scaled_data.shape}")

        # Áp dụng PCA lần 1 để xác định số PC tối ưu
        pca_full = PCA()
        pca_full.fit(scaled_data)
        explained_var_ratio = pca_full.explained_variance_ratio_
        eigenvalues = pca_full.explained_variance_
        explained = explained_var_ratio * 100
        cumulative = np.cumsum(explained)
            
        if method == "Elbow Method":
            knee = KneeLocator(
                x=range(1, len(explained_var_ratio) + 1),
                y=explained_var_ratio,
                curve='convex',
                direction='decreasing'
            )
            optimal_pc = knee.knee or len(explained_var_ratio)
            st.write(f"**Optimal PCs by Elbow method:** {optimal_pc}")
        else:
            # Kaiser
            optimal_pc = int((eigenvalues > 1).sum()) or 1
            st.write(f"**Optimal PCs by Kaiser criterion (eigenvalue >1):** {optimal_pc}")
        
        
        # Scree plot
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(range(1, len(explained_var_ratio) + 1), explained_var_ratio, 'o--')
        if method == "Elbow Method":
            ax1.axvline(optimal_pc, color='r', linestyle='--')
            ax1.set_xlabel("Number of Principal Components")
            ax1.set_ylabel("Explained Variance Ratio")
            ax1.set_title("Scree Plot")
            st.pyplot(fig1)

    # Áp dụng PCA lần 2 với số PC tối ưu
    pca_final = PCA(n_components=optimal_pc)
    pca_data = pca_final.fit_transform(scaled_data)
    st.write(f"**Data shape after PCA:** {pca_data.shape}")

    options = list(range(1, len(cumulative)+1))
    default = 2

    selected = st.multiselect(
        "Select how many PCs to report",
        options=options,
        default=default
    )

    st.subheader("Cumulative explained variance")
    for k in selected:
        st.write(f"- First **{k}** PCs explain **{cumulative[k-1]:.2f}%** of total variance")
    
    # Hiển thị độ đóng góp của từng biến vào các PC (PCA Loadings)
    loadings = pd.DataFrame(
        pca_final.components_.T,
        columns=[f'PC{i+1}' for i in range(optimal_pc)],
        index=numeric_cols
    )
    # Vẽ biểu đồ cột cho PCA Loadings
    with st.expander("View PCA Loadings (feature contributions)", expanded=False):
        st.dataframe(loadings)
        for pc in loadings.columns:
            st.write(f"**{pc}**")
            st.bar_chart(loadings[pc])

    # Tái tạo dữ liệu và Đánh giá lỗi
    reconstructed = pca_final.inverse_transform(pca_data)
    mse = mean_squared_error(scaled_data, reconstructed)
    st.write(f"**Reconstruction Error (MSE):** {mse:.5f}")

    # ---- VẼ PCA SCATTER PLOT cho PC1, PC2----
    per_var = np.round(pca_final.explained_variance_ratio_ * 100, 1)
    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(optimal_pc)])

    # Tìm cột phân loại (nếu có)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    category_col = next((c for c in cat_cols if df[c].nunique() <= 10), None)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    if category_col:
        for cat in df[category_col].astype(str).unique():
            idx = df[category_col] == cat
            ax2.scatter(pca_df.loc[idx, 'PC1'], pca_df.loc[idx, 'PC2'], label=cat, alpha=0.7)
        ax2.legend(title=category_col)
    else:
        ax2.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6)
    ax2.set_xlabel(f"PC1 ({per_var[0]}% variance)")
    ax2.set_ylabel(f"PC2 ({per_var[1]}% variance)")
    ax2.set_title("PCA Scatter Plot (PC1 vs PC2)")
    st.pyplot(fig2)

    # Download PCA đã giảm chiều
    pca_df_all = pd.concat([pca_df, df.reset_index(drop=True)], axis=1)
    csv = pca_df_all.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download PCA result as CSV",
        data=csv,
        file_name="pca_transformed.csv",
        mime='text/csv'
    )

