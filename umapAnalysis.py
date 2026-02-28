import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap

df = pd.read_csv("movies_2026_clean_scaled.csv")

numericCols = [
    "budget", "revenue", "runtime", "popularity",
    "voteAvg", "voteCount", "genresAmount",
    "productionCoAmount", "productionCountriesAmount",
    "actorsAmount", "castWomenAmount", "castMenAmount",
    "releaseYear"
]

X = df[numericCols].values

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)

X_umap = reducer.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_umap[:,0], X_umap[:,1], s=5, alpha=0.5)
plt.title("Proyección UMAP")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.show()