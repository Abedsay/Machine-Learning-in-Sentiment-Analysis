from flask import Flask, request, render_template
import pickle
import numpy as np

rf_model = pickle.load(open('rf_sentiment_model_tfidf.pkl', 'rb'))
lr_model = pickle.load(open('sentiment_model_tfidf.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_tfidf.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get review from form
    review = request.form.get('review')
    model_choice = request.form.get('model')  # RF or LR
    
    # Preprocess the review
    review_vectorized = vectorizer.transform([review]).toarray()
    
    # Predict sentiment based on chosen model
    if model_choice == 'RF':
        prediction = rf_model.predict(review_vectorized)
    elif model_choice == 'LR':
        prediction = lr_model.predict(review_vectorized)
    else:
        return render_template('index.html', pred="Invalid model selection.")
    
    # Format prediction
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return render_template('index.html', pred=f'Sentiment: {sentiment}')

if __name__ == '__main__':
    app.run(debug=True)
