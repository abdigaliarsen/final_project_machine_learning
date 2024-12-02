from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

data = pd.read_csv("../dataset/train.tsv", sep="\t")

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Phrase'])
sequences = tokenizer.texts_to_sequences(data['Phrase'])

max_length = 100
padded_sequences = pad_sequences(sequences, max_length, padding='post', truncating='post')

print(f"Padded shape: {padded_sequences.shape}")

