import os
import re
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# Get the project directory dynamically
project_directory = os.path.dirname(os.path.abspath(__file__))  # Get current script's directory
project_directory = os.path.join(project_directory, "..", "..")  # Go up two levels to reach the root (e.g., if this is inside 'notebooks')

# Define paths relative to the project directory
train_path = os.path.join(project_directory, "dataset", "train.tsv")
test_path = os.path.join(project_directory, "dataset", "test.tsv")
sample_submission_path = os.path.join(project_directory, "dataset", "sampleSubmission.csv")

# Define the output submission path
submission_path = os.path.join(project_directory, "submissions", "trainable_embedding_submission.csv")

# Load data
train_data = pd.read_csv(train_path, sep='\t')
test_data = pd.read_csv(test_path, sep='\t')

# Drop rows with NaN in 'Phrase' column in the test data
test_data = test_data.dropna(subset=['Phrase'])

# Preprocess the sentiment labels
Sentiment_phrase = []
for row in train_data['Sentiment']:
    if row == 0:
        Sentiment_phrase.append('negative')
    elif row == 1:
        Sentiment_phrase.append('somewhat negative')
    elif row == 2:
        Sentiment_phrase.append('neutral')
    elif row == 3:
        Sentiment_phrase.append('somewhat positive')
    elif row == 4:
        Sentiment_phrase.append('positive')
    else:
        Sentiment_phrase.append('Failed')

train_data['Sentiment_phrase'] = Sentiment_phrase

# Tokenize and preprocess text data (lowercase, remove stopwords, lemmatization)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize necessary components
stop_words = stopwords.words('english')
stop_words.remove('not')  # Keep 'not' because it's important in sentiment analysis
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer(language='english')

def data_preprocess(Phrase):
    # Remove HTML tags
    Phrase = re.sub(re.compile('<.*?>'), '', Phrase)
    # Remove non-alphabetic characters
    Phrase = re.sub('[^A-Za-z0-9]+', ' ', Phrase)
    # Convert to lowercase
    Phrase = Phrase.lower()
    # Tokenize text
    tokens = word_tokenize(Phrase)
    # Remove stopwords and apply lemmatization
    Phrase = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Rejoin tokens into a string
    Phrase = ' '.join(Phrase)

    return Phrase

train_data["Preprocessed_phrase"] = train_data['Phrase'].apply(data_preprocess)

# Tokenization and Padding
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['Preprocessed_phrase'])

train_sequences = tokenizer.texts_to_sequences(train_data['Preprocessed_phrase'])
test_sequences = tokenizer.texts_to_sequences(test_data['Phrase'])

max_length = 50  # Max length of sequences
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(train_data['Sentiment'], num_classes=5)
y_val = to_categorical(train_data['Sentiment'], num_classes=5)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(train_padded, y_train, test_size=0.2, random_state=42)

# Model definition
model = Sequential([
    Embedding(input_dim=10000, output_dim=50, input_length=max_length),
    Bidirectional(LSTM(128, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')  # 5 classes, using softmax activation for multi-class classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,  # Increased epochs
    batch_size=64  # Experiment with batch size
)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy:.2f}")

# Get predictions from the model
predictions = model.predict(test_padded)

# Convert predictions to the class label (0, 1, 2, 3, 4)
predicted_classes = predictions.argmax(axis=1)  # Get the class with the highest probability

# Create a Kaggle submission file
kaggle_submission = pd.read_csv(sample_submission_path)
kaggle_submission = kaggle_submission.head(len(test_data))  # Remove extra row if any

# Ensure that 'Sentiment' column is populated with the predicted labels
kaggle_submission['Sentiment'] = predicted_classes

# Save the submissions to a CSV file
kaggle_submission.to_csv(submission_path, index=False)

# Debug: Confirm file has been saved
print(f"Submission saved to {submission_path}")
