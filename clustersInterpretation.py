import pandas as pd
from sklearn.cluster import KMeans

df_original = pd.read_csv("movies_2026_clean.csv")

df_scaled = pd.read_csv("movies_2026_clean_scaled.csv")

numericCols = [
    "budget", "revenue", "runtime", "popularity",
    "voteAvg", "voteCount", "genresAmount",
    "productionCoAmount", "productionCountriesAmount",
    "actorsAmount", "castWomenAmount", "castMenAmount",
    "releaseYear"
]

X = df_scaled[numericCols].values

K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

df_original["cluster"] = labels

print("\nMedias por cluster (variables numéricas):\n")
print(df_original.groupby("cluster")[numericCols].mean())

print("\nMedianas por cluster:\n")
print(df_original.groupby("cluster")[numericCols].median())

print("\nFrecuencia de idiomas por cluster:\n")
print(df_original.groupby("cluster")["originalLanguage"].value_counts().head(20))

print("\nFrecuencia de géneros por cluster:\n")
print(df_original.groupby("cluster")["genres"].value_counts().head(20))