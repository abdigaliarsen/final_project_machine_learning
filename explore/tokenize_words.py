from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

data = pd.read_csv("../dataset/train.tsv", sep="\t")

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Phrase'])
sequences = tokenizer.texts_to_sequences(data['Phrase'])
word_index = tokenizer.word_index

print(f"Vocabulary size: {len(word_index)}")
print(f"Sample sequence: {sequences[0]}")
