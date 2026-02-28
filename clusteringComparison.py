import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
  
np.random.seed(42)

df = pd.read_csv("movies_2026_clean_scaled.csv")

numericCols = [
    "budget", "revenue", "runtime", "popularity",
    "voteAvg", "voteCount", "genresAmount",
    "productionCoAmount", "productionCountriesAmount",
    "actorsAmount", "castWomenAmount", "castMenAmount",
    "releaseYear"
]

X = df[numericCols].values

K = 4

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X)

sil_kmeans = silhouette_score(X, labels_kmeans)

hierarchical = AgglomerativeClustering(n_clusters=K, linkage='ward')
labels_hier = hierarchical.fit_predict(X)

sil_hier = silhouette_score(X, labels_hier)

print("===================================")
print("RESULTADOS DE CLUSTERING")
print("===================================")
print(f"K seleccionado: {K}")
print("-----------------------------------")
print(f"Silhouette K-Means: {sil_kmeans:.4f}")
print(f"Silhouette Jerárquico: {sil_hier:.4f}")

print("\nTamaño de clusters (K-Means):")
print(pd.Series(labels_kmeans).value_counts().sort_index())

print("\nTamaño de clusters (Jerárquico):")
print(pd.Series(labels_hier).value_counts().sort_index())