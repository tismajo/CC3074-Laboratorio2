
"""
Exploración de los datos.
"""
import pandas as pd

#df = pd.read_csv("movies_2026.csv", encoding="latin-1")
df = pd.read_csv("movies_2026_clean.csv", encoding="latin-1")

nullsCount = df.isnull().sum()
summary = pd.DataFrame({
    "nulos": nullsCount,
    "no_nulos": df.notnull().sum(),
    "porcentaje_nulos": df.isnull().mean() * 100,
    "tipo_dato": df.dtypes
}).sort_values(by="porcentaje_nulos", ascending=False)
rowColumns = df.shape
columnsTypes = df.dtypes
#duplicatedCount = df.duplicated().sum()
firstFive = df.head(5)

print(f"=================\n(filas, columnas)\n=================\n{rowColumns}")
print(f"\n=======================\nCantidad de datos nulos\n=======================\n{summary}")
print(f"\n================\nTipo por columna\n================\n{columnsTypes}")
#print(f"\n================\nDatos duplicados\n================\n{duplicatedCount}")
print(f"\n==============\nPrimeros datos\n==============\n{firstFive}")
print("\n=======================\nDescripción estadística\nvariables numéricas\n=======================\n")
print(df[["budget", "revenue"]].describe())
print("\n==============\nCantidad de 0s\n==============\n")
print((df["budget"] == 0).sum())
print((df["revenue"] == 0).sum())
