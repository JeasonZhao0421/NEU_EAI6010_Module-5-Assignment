import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import joblib

# Load data in chunks and combine into a single DataFrame
train_df = pd.concat([chunk for chunk in pd.read_csv('Corona_NLP_train.csv', encoding='ISO-8859-1', chunksize=1000)])
test_df = pd.concat([chunk for chunk in pd.read_csv('Corona_NLP_test.csv', encoding='ISO-8859-1', chunksize=1000)])

# Encode the 'Sentiment' column as numeric labels for training
train_df['Sentiment'] = train_df['Sentiment'].astype('category').cat.codes
test_df['Sentiment'] = test_df['Sentiment'].astype('category').cat.codes

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(train_df['OriginalTweet'])  # Transform training text
X_test_tfidf = vectorizer.transform(test_df['OriginalTweet'])        # Transform test text

# Train a Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, train_df['Sentiment'])

joblib.dump(nb_model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

nb_model = joblib.load('/path/to/model.pkl')
vectorizer = joblib.load('/path/to/vectorizer.pkl')

# Flask application for serving predictions
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint for predicting sentiment.
    Expects JSON input with a 'text' field.
    Returns predicted sentiment as a JSON response.
    """
    data = request.json
    text = data.get("text", "")
    text_vectorized = vectorizer.transform([text])  # Vectorize input text
    prediction = nb_model.predict(text_vectorized)[0]  # Predict sentiment
    return jsonify({"sentiment": int(prediction)})

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Sentiment Analysis API! Use /predict to get predictions."

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == "__main__":
    # Run the Flask app on port 5000
    app.run(host="0.0.0.0", port=5000)
