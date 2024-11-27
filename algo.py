import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the CSV file
file_path = 'StudentPerformanceFactors.csv'  # Use the correct path to the file
df = pd.read_csv(file_path)

# Inspect the first few rows to understand the structure of the data
print("Data Preview:")
print(df.head())

# Step 1: Preprocessing the Data
# Convert numerical features to categorical bins
df['Hours_Studied_Bin'] = pd.cut(df['Hours_Studied'], bins=[0, 20, 30, 40], labels=['Low', 'Medium', 'High'])
df['Attendance_Bin'] = pd.cut(df['Attendance'], bins=[0, 60, 80, 100], labels=['Low', 'Medium', 'High'])

# One-hot encode categorical features
df_encoded = pd.get_dummies(df[['Hours_Studied_Bin', 'Attendance_Bin', 'Parental_Involvement', 'Distance_from_Home', 'Gender']], drop_first=True)

# If 'Exam_Score' is important, you may need to bin it as well
df_encoded['Exam_Score_Bin'] = pd.cut(df['Exam_Score'], bins=[0, 60, 75, 100], labels=['Low', 'Medium', 'High'])
df_encoded = pd.get_dummies(df_encoded, columns=['Exam_Score_Bin'], drop_first=True)

# Step 2: Apply the Apriori Algorithm
min_support = 0.5  # Example: at least 50% of transactions
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# Step 3: Generate Association Rules
min_confidence = 0.7  # Example: at least 70% confidence
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Display the frequent itemsets and the association rules
print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)
