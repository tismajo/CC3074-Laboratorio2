import pandas as pd
import numpy as np
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

df = pd.read_csv("movies_2026_clean_scaled.csv")

numericCols = [
    "budget", "revenue", "runtime", "popularity",
    "voteAvg", "voteCount", "genresAmount",
    "productionCoAmount", "productionCountriesAmount",
    "actorsAmount", "castWomenAmount", "castMenAmount",
    "releaseYear"
]

X = df[numericCols].dropna()

kmo_all, kmo_model = calculate_kmo(X)

print("===================================")
print("Índice KMO global:", round(kmo_model, 4))
print("===================================")

chi_square_value, p_value = calculate_bartlett_sphericity(X)

print("\nTest de Esfericidad de Bartlett")
print("Chi-cuadrado:", round(chi_square_value, 4))
print("p-valor:", p_value)