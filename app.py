import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and vectorizer
model_path = "uploads/model.pkl"  # Replace with the actual path to your model
vectorizer_path = "uploads/vectorizer.pkl"  # Replace with the actual path to your vectorizer

# Load the model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load the vectorizer
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict sentiment.
    Expects a JSON payload with a 'text' key.
    """
    try:
        # Get the text from the POST request
        data = request.get_json()
        input_text = data.get('text', '')

        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        # Vectorize the input text
        X = vectorizer.transform([input_text])

        # Predict sentiment
        prediction = model.predict(X)

        # Return the result as JSON
        response = {
            "input_text": input_text,
            "predicted_label": int(prediction[0])  # Convert numpy.int64 to int
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)
