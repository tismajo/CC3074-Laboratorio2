import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("movies_2026.csv", encoding="latin-1")

numericCols = [
    "budget", "revenue", "runtime", "popularity",
    "voteAvg", "voteCount", "genresAmount",
    "productionCoAmount", "productionCountriesAmount",
    "actorsAmount", "castWomenAmount", "castMenAmount",
    "releaseYear"
]

df = df.drop(columns=[
    "homePage", 
    "actorsCharacter", 
    "originalTitle", 
    "title", 
    "id"
])

df['productionCompany'] = df['productionCompany'].fillna('Unknown')
df['productionCountry'] = df['productionCountry'].fillna('Unknown')
df['genres'] = df['genres'].fillna('Unknown')
df['director'] = df['director'].fillna('Unknown')

df['castMenAmount'] = df['castMenAmount'].fillna(df['castMenAmount'].median())
df['castWomenAmount'] = df['castWomenAmount'].fillna(df['castWomenAmount'].median())

df = df.dropna(subset=numericCols)

scaler = StandardScaler()
df_scaled = df.copy()

df_scaled[numericCols] = scaler.fit_transform(df[numericCols])

df.to_csv("movies_2026_clean.csv", index=False)
df_scaled.to_csv("movies_2026_clean_scaled.csv", index=False)

print("Limpieza completada.")
print("Archivo limpio guardado como movies_2026_clean.csv")
print("Archivo normalizado guardado como movies_2026_clean_scaled.csv")
