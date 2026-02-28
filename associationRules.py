import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("movies_2026_clean.csv")

# =========================
# Discretización robusta
# =========================

df["budget_cat"] = pd.cut(
    df["budget"],
    bins=[-1, 0, 10000000, 50000000, df["budget"].max()],
    labels=["ZeroBudget", "LowBudget", "MidBudget", "HighBudget"]
)

df["revenue_cat"] = pd.cut(
    df["revenue"],
    bins=[-1, 0, 10000000, 100000000, df["revenue"].max()],
    labels=["ZeroRevenue", "LowRevenue", "MidRevenue", "HighRevenue"]
)

df["runtime_cat"] = pd.cut(
    df["runtime"],
    bins=[0, 40, 90, 120, df["runtime"].max()],
    labels=["Short", "Medium", "Long", "VeryLong"]
)
df["vote_cat"] = pd.cut(
    df["voteAvg"],
    bins=[0, 4, 6, 8, 10],
    labels=["LowVote", "MidVote", "HighVote", "VeryHighVote"]
)

df["genres_split"] = df["genres"].apply(lambda x: x.split("|"))

transactions = []

for _, row in df.iterrows():
    transaction = []
    transaction.extend(row["genres_split"])

    transaction.append("Lang_" + str(row["originalLanguage"]))

    transaction.append(str(row["budget_cat"]))
    transaction.append(str(row["revenue_cat"]))
    transaction.append(str(row["runtime_cat"]))
    transaction.append(str(row["vote_cat"]))
    
    transactions.append(transaction)

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

rules = rules.sort_values(by="lift", ascending=False)

print("\nReglas encontradas:\n")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(20))
