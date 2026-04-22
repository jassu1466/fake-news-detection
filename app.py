from flask import Flask, request, render_template
import pickle
import string

app = Flask(__name__)

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text function
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)

    if result[0] == 1:
        prediction = "🟢 Real News"
    else:
        prediction = "🔴 Fake News"

    return render_template('index.html', prediction=prediction)

# Run app
if __name__ == "__main__":
    app.run(debug=True)