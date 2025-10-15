# -----------------------------------------------------------------------------
# Market Basket Analysis with Apriori
# -----------------------------------------------------------------------------
# This script performs a market basket analysis on a sample dataset of
# transactions to identify frequent itemsets and generate association rules.
#
# Libraries: pandas, mlxtend, matplotlib
# -----------------------------------------------------------------------------

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

def main():
    """
    Main function to run the entire market basket analysis pipeline.
    """
    # --- 1. Data Preparation ---
    # Sample transaction data. In a real-world scenario, this would be loaded
    # from a file or a database (e.g., a CSV of sales records).
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

    # Transform the transaction data into a one-hot encoded DataFrame.
    # This format is required by the Apriori algorithm.
    encoder = TransactionEncoder()
    one_hot_encoded_array = encoder.fit(transactions).transform(transactions)
    df = pd.DataFrame(one_hot_encoded_array, columns=encoder.columns_)

    print("--- One-Hot Encoded Transaction DataFrame ---")
    print(df)

    # --- 2. Frequent Itemset Mining ---
    # Apply the Apriori algorithm to find itemsets with a support of at least 50%.
    # Support is the proportion of transactions in which an itemset appears.
    frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

    print("\n--- Frequent Itemsets (Support >= 0.5) ---")
    print(frequent_itemsets)

    # --- 3. Association Rule Generation ---
    # Generate association rules from the frequent itemsets.
    # We are looking for rules with a confidence of at least 70%.
    # Confidence is a measure of the reliability of the rule.
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

    # Sort the rules by confidence and lift for better readability.
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

    print("\n--- Discovered Association Rules (Confidence >= 0.7) ---")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string())

    # --- 4. Visualization of Top Items ---
    # Calculate the frequency of each individual item.
    item_counts = df.sum().sort_values(ascending=False)
    top_5_items = item_counts.head(5)

    # Create a bar chart to visualize the most frequent items.
    plt.figure(figsize=(10, 6))
    top_5_items.plot(kind='bar', color='c')
    plt.title('Top 5 Most Frequent Items in Transactions', fontsize=16)
    plt.xlabel('Items', fontsize=12)
    plt.ylabel('Number of Transactions', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # --- 5. Save Results ---
    # Save the generated association rules to a CSV file for further analysis.
    output_file = 'market_basket_rules.csv'
    rules.to_csv(output_file, index=False)

    print(f"\nSuccessfully saved association rules to '{output_file}'")

if __name__ == "__main__":
    main()
