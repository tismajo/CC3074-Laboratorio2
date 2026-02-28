# Laboratorio 2 – Clustering. A priori. PCA.

**CC3074 – Minería de Datos**
*Universidad del Valle de Guatemala*

## Descripción del Proyecto
Este proyecto analiza un conjunto de datos de 19,883 películas obtenidas de The Movie Database (TMDB) utilizando técnicas de aprendizaje no supervisado.

El objetivo principal es identificar patrones estructurales en el mercado cinematográfico que permitan generar insights estratégicos para la empresa ficticia CineVision Studios.

Se aplicaron los siguientes métodos:

- Clustering (K-Means y Jerárquico)
- Estadístico de Hopkins
- VAT (Visual Assessment of Cluster Tendency)
- Reglas de Asociación (Apriori)
- Análisis de Componentes Principales (PCA)
- UMAP (reducción de dimensionalidad no lineal)

## Estructura del Repositorio
.
├── cleanData.py
├── dataExploration.py
├── hopkins.py
├── vatAnalysis.py
├── elbow_method.py
├── clustering_comparison.py
├── clustersInterpretation.py
├── associationRules.py
├── pcaPrecheck.py
├── pcaAnalysis.py
├── umap_analysis.py
├── movies_2026_clean.csv
├── movies_2026_clean_scaled.csv
└── README.md
## 1. Limpieza y Preprocesamiento
- Se realizaron las siguientes tareas:
- Eliminación de columnas irrelevantes (ID, títulos duplicados, homepage, etc.).
- Imputación de valores faltantes en variables categóricas.
- Imputación de valores numéricos utilizando la mediana.
- Estandarización de variables numéricas mediante StandardScaler.
- Se generaron dos datasets:
- movies_2026_clean.csv: datos limpios.
- movies_2026_clean_scaled.csv: datos normalizados para modelado.
## 2. Tendencia al Agrupamiento
Estadístico de Hopkins
Resultado obtenido:
`H = 0.9933`
Interpretación:
El conjunto de datos presenta una estructura altamente no uniforme, lo que indica fuerte tendencia al agrupamiento.
Sin embargo, esta estructura está dominada por concentración económica (muchos valores de presupuesto e ingresos en cero).

**VAT (Visual Assessment of Cluster Tendency)**
- Se observó una gran masa central con algunos puntos periféricos.
- No se identificaron bloques perfectamente separados.
- La estructura es gradual más que categórica.
*Conclusión:*
Existe estructura, pero no clusters rígidos claramente delimitados.

## 3. Clustering
**Método del Codo**
- No se observó un codo pronunciado.
- La curva mostró reducción progresiva de inercia.
- Se seleccionó K = 4 como compromiso razonable.

**Comparación de Algoritmos**
Método	Silhouette
K-Means	0.3664
Jerárquico (Ward)	0.3371

Interpretación:
Separación moderada.
K-Means mostró mejor desempeño.
Se eligió K-Means para interpretación final.

**Interpretación de Clusters**
Se identificaron:
- Un cluster dominante de producciones de bajo presupuesto.
- Un cluster pequeño de superproducciones (alto presupuesto y alto ingreso).
- Un cluster asociado a producciones muy recientes y cortas.
- Un cluster intermedio con características comerciales estándar.

*Conclusión:*
El mercado cinematográfico está organizado principalmente por escala económica.

## 4. Reglas de Asociación (Apriori)
Se discretizaron variables numéricas y se aplicó Apriori con distintos niveles de soporte y confianza.
Hallazgos relevantes:
- Películas de alto ingreso están fuertemente asociadas con idioma inglés y alta votación.
- Presupuesto medio también se asocia con buena recepción crítica.
- Se eliminaron categorías dominantes (nan, Unknown) para evitar reglas triviales.

*Conclusión:*
El éxito económico está significativamente asociado con idioma y valoración del público.

## 5. Análisis de Componentes Principales (PCA)
**Pruebas previas**
- KMO = 0.7884 → Buena adecuación.
- Bartlett p < 0.05 → Correlaciones significativas.
- PCA es estadísticamente justificable.

**Resultados**
- PC1 explica 28.85% de la varianza.
- Los primeros 4 componentes explican 60.58%.
- Los primeros 6 explican 75.47%.

Interpretación:
PC1 representa una dimensión económica dominante (budget, revenue, voteCount).
Componentes posteriores capturan estructura productiva y composición del elenco.

*Conclusión:*
La variabilidad del dataset está dominada por factores económicos.

## 6. UMAP (Reducción No Lineal)
UMAP permitió visualizar la estructura en 2D.
Observaciones:
- Masa central continua.
- Regiones periféricas asociadas a producciones extremas.
- No existen clusters completamente separados.

*Conclusión:*
El mercado cinematográfico funciona como un continuo económico, no como segmentos rígidos.

## Conclusiones Generales

1. El dataset presenta fuerte estructura, pero gradual.
2. La dimensión económica domina la organización del mercado.
3. No existen segmentos discretos claramente definidos.
4. Las superproducciones constituyen una minoría estructuralmente distinta.
5. El idioma inglés está fuertemente asociado al éxito comercial.

## Requisitos
```shell
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend factor_analyzer umap-learn
```