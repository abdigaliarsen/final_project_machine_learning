import pandas as pd

data = pd.read_csv('../dataset/train.tsv', sep='\t')

print("FIRST 5 DATASET ROWS")
print(data.head(), end="\n\n------\n")

print("DATASET INFO")
print(data.info(), end="\n\n------\n")

print("SENTIMENT VALUES COUNT")
print(data['Sentiment'].value_counts(), end="\n\n------\n")
