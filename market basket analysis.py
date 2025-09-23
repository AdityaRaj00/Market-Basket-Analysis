import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

transactions = [
    ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
    ['Milk', 'Onion', 'Kidney Beans', 'Eggs', 'Ice Cream'],
    ['Apple', 'Chicken', 'Eggs', 'Yogurt'],
    ['Onion', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Milk', 'Chicken', 'Eggs', 'Yogurt'],
    ['Apple', 'Milk', 'Chicken', 'Eggs'],
]

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

print("--- One-Hot Encoded DataFrame ---")
print(df)

frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

print("\n--- Frequent Itemsets (Support >= 0.5) ---")
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

print("\n--- Discovered Association Rules (Confidence >= 0.7) ---")
print(rules)

item_counts = df.sum().sort_values(ascending=False)
top_5_items = item_counts.head(5)

plt.figure(figsize=(8, 5))
top_5_items.plot(kind='bar', color='skyblue')
plt.title('Top 5 Most Frequent Items')
plt.xlabel('Items')
plt.ylabel('Transaction Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

output_file = 'market_basket_analysis_rules.csv'
rules.to_csv(output_file, index=False)

print(f"\nSuccessfully saved association rules to '{output_file}'")
