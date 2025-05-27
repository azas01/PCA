import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from kneed import KneeLocator

# Load dataset
file_path = input("Enter the file path: ")
df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

# Chỉ lấy các cột số
numeric_cols = df.select_dtypes(include=[np.number]).columns
data_numeric = df[numeric_cols]

# Chuẩn hóa dữ liệu 
scaled_data = preprocessing.scale(data_numeric)
print("Original Shape:", scaled_data.shape)

# Áp dụng PCA lần 1 để xác định số PC tối ưu
pca_full = PCA()
pca_full.fit(scaled_data)

# Tính phương sai
explained_variance = pca_full.explained_variance_ratio_

# Xác định "góc gãy" (elbow point)
knee_locator = KneeLocator(range(1, len(explained_variance) + 1), explained_variance, curve="convex", direction="decreasing")
optimal_pc = knee_locator.knee

print(f"Optimal number of main PCs by Elbow method: {optimal_pc}")

# Vẽ Scree Plot với điểm tối ưu
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o--', label="Explained Variance")
plt.axvline(optimal_pc, color='r', linestyle='--', label=f"Optimal PC = {optimal_pc}")
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot - Elbow Method")
plt.legend()
plt.tight_layout()
plt.show()

# Áp dụng PCA lần 2 với số PC tối ưu
pca_final = PCA(n_components=optimal_pc)
pca_data = pca_final.fit_transform(scaled_data)

print(f"Shape after PCA reduction: {pca_data.shape}")

# Hiển thị độ đóng góp của từng biến vào các PC (PCA Loadings)
loadings = pd.DataFrame(pca_final.components_.T,
                        columns=[f'PC{i+1}' for i in range(optimal_pc)],
                        index=numeric_cols)
print("\nPCA Loadings (feature contributions to each PC):")
print(loadings)

# Vẽ biểu đồ cột cho PCA Loadings
for i in range(optimal_pc):
    pc_label = f'PC{i+1}'
    plt.figure(figsize=(10, 4))
    sns.barplot(x=loadings.index, y=loadings[pc_label])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Feature Contributions to {pc_label}')
    plt.ylabel('Loading Value')
    plt.xlabel('Original Features')
    plt.tight_layout()
    plt.show()

# Tái tạo dữ liệu và Đánh giá lỗi
reconstructed_data = pca_final.inverse_transform(pca_data)
error = mean_squared_error(scaled_data, reconstructed_data)
print(f"Reconstruction Error (MSE): {error}")

# ---- VẼ PCA SCATTER PLOT ----
per_var = np.round(pca_final.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
pca_df = pd.DataFrame(pca_data, columns=labels)

# Tìm cột phân loại (nếu có)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
category_col = None

for col in categorical_cols:
    if df[col].nunique() <= 10:  
        category_col = col
        break

plt.figure(figsize=(8, 6))

if category_col:
    pca_df[category_col] = df[category_col]
    unique_categories = df[category_col].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_categories)))
    category_color_map = dict(zip(unique_categories, colors))

    for category in unique_categories:
        subset = pca_df[pca_df[category_col] == category]
        plt.scatter(subset['PC1'], subset['PC2'], label=category, color=category_color_map[category])

    plt.legend(title=category_col)
else:
    plt.scatter(pca_df['PC1'], pca_df['PC2'], color='blue', alpha=0.5)

plt.title('PCA Scatter Plot (PC1 vs PC2)')
plt.xlabel(f'PC1 - {per_var[0]}% variance')
plt.ylabel(f'PC2 - {per_var[1]}% variance')
plt.tight_layout()
plt.show()

# Lưu dữ liệu PCA đã giảm chiều
output_path = file_path.replace(".csv", "_pca.csv") if file_path.endswith(".csv") else file_path.replace(".xlsx", "_pca.xlsx")
pca_df.to_csv(output_path, index=False)
print(f"Reduced data saved to: {output_path}")
