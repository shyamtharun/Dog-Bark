from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import joblib

from main import extract_features, predict_emotion, train_model, load_dataset  # Adjust import as needed

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load or train your model
X, y = load_dataset('audio', 'labels')
model = train_model(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return "No file part", 400

    file = request.files['audio_file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        try:
            emotion = predict_emotion(model, filepath)
            return jsonify({'emotion': emotion})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
