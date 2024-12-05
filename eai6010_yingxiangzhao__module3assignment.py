import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import joblib
import os

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
from flask import Flask, request, jsonify
import joblib

# Flask application for serving predictions
app = Flask(__name__)

nb_model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 获取请求中的 JSON 数据
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid input. 'text' field is required."}), 400

        # 获取文本并进行向量化
        sample_text = [data["text"]]
        vectorized_text = vectorizer.transform(sample_text)

        # 预测并返回结果
        prediction = nb_model.predict(vectorized_text)[0]
        return jsonify({"sentiment": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Sentiment Analysis API! Use /predict to get predictions."

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app on port 5000
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

