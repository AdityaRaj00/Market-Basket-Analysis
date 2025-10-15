# 🛒 Market Basket Analysis using Apriori Algorithm

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)
![Output](https://img.shields.io/badge/Output-CSV-lightgrey?logo=file)

This project demonstrates a classic **Market Basket Analysis** using Python to uncover purchasing patterns from a sample set of transactions.  
It utilizes the **Apriori algorithm** from the `mlxtend` library to identify frequent itemsets and derive association rules — revealing which products are likely to be purchased together.


## 🎯 Analysis Overview

The goal of Market Basket Analysis is to find **relationships between items**.  
This is the same technique used by retailers like **Amazon** and **grocery stores** for:

- Product recommendations (“Customers who bought this also bought…”)
- Strategic product placement and bundling

This script performs the following steps:

1. **Data Transformation:**  
   Converts the raw list of transactions into a one-hot encoded DataFrame suitable for analysis.

2. **Frequent Itemset Mining:**  
   Applies the Apriori algorithm to find groups of items that appear together frequently (based on a `min_support` threshold).

3. **Association Rule Generation:**  
   Creates rules from the frequent itemsets that predict the likelihood of buying one product given another (based on a `min_confidence` threshold).

4. **Visualization:**  
   Generates a bar chart to show the most popular individual items in the dataset.

5. **Output:**  
   Saves the discovered association rules to a CSV file for further review.

---

## 📊 Key Metrics Explained

| Metric | Description | Example |
|--------|--------------|----------|
| **Support** | Fraction of transactions that contain a specific itemset. | A support of 0.5 for `{Milk, Eggs}` means they appear together in 50% of transactions. |
| **Confidence** | Likelihood that a customer will buy item B given they bought item A. | A confidence of 0.8 for `{Milk} → {Eggs}` means 80% of Milk buyers also bought Eggs. |
| **Lift** | Measures how much more likely item B is purchased when item A is purchased. Lift > 1 implies a positive relationship. | A lift of 1.5 means buying Milk makes Eggs 1.5× more likely to be bought. |

---

## 🧠 Prerequisites

- **Python 3.7+**
- Required Python Libraries:
  ```bash
        pip install pandas mlxtend matplotlib
## ▶️ How to Run
Clone or download this repository.

Ensure all required libraries are installed.

Open a terminal and navigate to the project directory.

Run the script:

        python analysis.py
## 📈 Results
After execution, the script will:

Print the one-hot encoded data, frequent itemsets, and the generated association rules to the console.

Display a bar chart of the top 5 most frequent items.

Create a CSV file named:

        market_basket_rules.csv
containing the detailed association rules.

## 🧱 Tech Stack
|Component|Technology|
|----- |-----|
|**Programming Language**|Python|
|**Libraries Used**|	Pandas, mlxtend, Matplotlib|
|**Algorithm**|Apriori|
|**Output Format**|CSV + Visualization|
