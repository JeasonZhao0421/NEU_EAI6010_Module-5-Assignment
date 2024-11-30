# -*- coding: utf-8 -*-
"""EAI6010_YingxiangZhao_ Module3Assignment.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10e1djKMFZjUSiSqN3hirdNE-3aOX1KPI
"""

# Install required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load train and test data with specified encoding
for chunk in pd.read_csv('Corona_NLP_train.csv', encoding='ISO-8859-1', chunksize=1000):
    process_chunk(chunk)
for chunk in pd.read_csv('Corona_NLP_test.csv', encoding='ISO-8859-1', chunksize=1000):
    process_chunk(chunk)

import gc
del train_df, test_df
gc.collect()

# Display the first few rows of the dataset
train_df.head()

# Encode the Sentiment labels as numbers
train_df['Sentiment'] = train_df['Sentiment'].astype('category').cat.codes
test_df['Sentiment'] = test_df['Sentiment'].astype('category').cat.codes

# Split data for training and validation
train_texts, valid_texts, train_labels, valid_labels = train_test_split(
    train_df['OriginalTweet'], train_df['Sentiment'], test_size=0.2, random_state=42)

# Prepare data for ULMFiT
dls = TextDataLoaders.from_df(train_df, text_col='OriginalTweet', label_col='Sentiment', valid_pct=0.2)

# Define the ULMFiT model using AWD_LSTM
learn = text_classifier_learner(dls, AWD_LSTM, metrics=accuracy)
learn.fine_tune(4)  # Fine-tune the model with 4 epochs

# Evaluate on test set
test_dl = learn.dls.test_dl(test_df['OriginalTweet'])
preds, _ = learn.get_preds(dl=test_dl)
test_df['Predicted_ULMFiT'] = preds.argmax(dim=1)

# Calculate accuracy for ULMFiT
ulmfit_accuracy = accuracy_score(test_df['Sentiment'], test_df['Predicted_ULMFiT'])
print(f"ULMFiT Model Accuracy: {ulmfit_accuracy}")

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(train_texts)
X_valid_tfidf = vectorizer.transform(valid_texts)
X_test_tfidf = vectorizer.transform(test_df['OriginalTweet'])

# Train Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, train_labels)

# Predict on test data
nb_preds = nb_model.predict(X_test_tfidf)
test_df['Predicted_NaiveBayes'] = nb_preds

# Calculate accuracy for Naive Bayes
nb_accuracy = accuracy_score(test_df['Sentiment'], test_df['Predicted_NaiveBayes'])
print(f"Naive Bayes Model Accuracy: {nb_accuracy}")

# ULMFiT classification report
print("ULMFiT Model Classification Report")
print(classification_report(test_df['Sentiment'], test_df['Predicted_ULMFiT']))

# Naive Bayes classification report
print("Naive Bayes Model Classification Report")
print(classification_report(test_df['Sentiment'], test_df['Predicted_NaiveBayes']))

# Plot accuracies
plt.figure(figsize=(8, 5))
plt.bar(['ULMFiT', 'Naive Bayes'], [ulmfit_accuracy, nb_accuracy], color=['blue', 'green'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison between ULMFiT and Naive Bayes')
plt.show()

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    return jsonify({"sentiment": "Positive"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
