import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list

np.random.seed(42)

df = pd.read_csv("movies_2026_clean_scaled.csv")

numericCols = [
    "budget", "revenue", "runtime", "popularity",
    "voteAvg", "voteCount", "genresAmount",
    "productionCoAmount", "productionCountriesAmount",
    "actorsAmount", "castWomenAmount", "castMenAmount",
    "releaseYear"
]

X = df[numericCols].dropna().values

sample_size = 500

if sample_size > X.shape[0]:
    sample_size = int(0.1 * X.shape[0])

indices = np.random.choice(X.shape[0], sample_size, replace=False)
X_sample = X[indices]

# =========================
# Calcular matriz de distancias
# =========================
dist_matrix = squareform(pdist(X_sample, metric='euclidean'))

# =========================
# Reordenar usando clustering jerárquico
# =========================
Z = linkage(X_sample, method='ward')

order = leaves_list(Z)

ordered_dist_matrix = dist_matrix[order][:, order]

# =========================
# Graficar VAT reordenado
# =========================
plt.figure(figsize=(8, 6))
sns.heatmap(ordered_dist_matrix, cmap='viridis')
plt.title("VAT Reordenado (Jerárquico)")
plt.xlabel("Observaciones")
plt.ylabel("Observaciones")
plt.tight_layout()
plt.show()
