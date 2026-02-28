import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("movies_2026_clean_scaled.csv")

numericCols = [
    "budget", "revenue", "runtime", "popularity",
    "voteAvg", "voteCount", "genresAmount",
    "productionCoAmount", "productionCountriesAmount",
    "actorsAmount", "castWomenAmount", "castMenAmount",
    "releaseYear"
]

X = df[numericCols].dropna().values

pca = PCA()
X_pca = pca.fit_transform(X)

explained_variance = pca.explained_variance_ratio_

print("\nVarianza explicada por componente:\n")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.4f}")

print("\nVarianza acumulada:\n")
print(np.cumsum(explained_variance))

plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
plt.xlabel("Componente Principal")
plt.ylabel("Varianza Explicada")
plt.title("Scree Plot")
plt.grid(True)
plt.show()

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(numericCols))],
    index=numericCols
)

print("\nCargas de los componentes:\n")
print(loadings.round(3))