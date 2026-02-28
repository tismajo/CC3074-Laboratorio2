import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

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

def hopkins(X, m=500):
    n, d = X.shape

    if m > n:
        m = int(0.1 * n)

    nbrs = NearestNeighbors(n_neighbors=2)
    nbrs.fit(X)

    rand_indices = np.random.choice(n, m, replace=False)
    X_sample = X[rand_indices]

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    U = np.random.uniform(mins, maxs, (m, d))

    w_dist, _ = nbrs.kneighbors(X_sample, n_neighbors=2)
    w = w_dist[:, 1]

    u_dist, _ = nbrs.kneighbors(U, n_neighbors=1)
    u = u_dist[:, 0]

    H = np.sum(u) / (np.sum(u) + np.sum(w))
    return H

H = hopkins(X, m=500)

print("===================================")
print(f"Estadístico de Hopkins: {H:.4f}")
print("===================================")
