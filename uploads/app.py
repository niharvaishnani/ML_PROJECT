# from flask import Flask, request, jsonify
# #from flask_cors import CORS
# import joblib
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder


# app = Flask(__name__)

# # Load your trained model
# model = joblib.load('ipl.pkl')
# #CORS(app)  # Enable CORS for all routes

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json  # Get JSON data sent in the POST request
    
#     df = pd.DataFrame([data])  
    
#     demo2 = {}
#     for column in ['batting_team', 'bowling_team', 'city']:
#         le = LabelEncoder()
#         df[column] = le.fit_transform(df[column])
#         demo2[column] = le  # Save the encoder for future use

#     prediction = model.predict(df)

#     return jsonify({'prediction': prediction.tolist()}) 

# #if __name__ == '__main__':
#     app.run()




























from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
model_path = "D:/SEM 6/MACHINE LEARNING/Project/uploads/model.pkl"  # Replace with the actual path to your model
vectorizer_path = "D:/SEM 6/MACHINE LEARNING/Project/uploads/vectorizer.pkl"  # Replace with the actual path to your vectorizer

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
    app.run(debug=True)

# 0.irrelevant
# 1.Negative 
# 2.Netural
# 3.Positive