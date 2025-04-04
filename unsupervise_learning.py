# 📦 Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer


df = pd.read_csv(r"C:\Users\Hello\Downloads\Data science notes\E-commerce_DataScience_project\project\amazon_products.csv")

print(df.head())

features = ['Price', 'Rating', 'Reviews']


df_cleaned = df.dropna(subset=features).copy()

for col in features:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')


df_cleaned.dropna(subset=features, inplace=True)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cleaned[features])


model = KMeans(n_init=10, random_state=42)
visualizer = KElbowVisualizer(model, k=(2, 10), metric='distortion')
visualizer.fit(df_scaled) 
visualizer.show()


optimal_k = visualizer.elbow_value_
print(f"✅ Optimal number of clusters: {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_cleaned['Cluster'] = kmeans.fit_predict(df_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_cleaned[features[0]], y=df_cleaned[features[1]], 
                hue=df_cleaned['Cluster'], palette='viridis', alpha=0.7)
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('🛒 Product Clustering (K-Means)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

output_path = "amazon_products_clustered.csv"
df_cleaned.to_csv(output_path, index=False)
print(f"Clustered dataset saved as '{output_path}'")


