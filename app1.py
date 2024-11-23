from flask import Flask, request, jsonify, redirect, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from io import BytesIO
from PIL import Image
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS  # Import CORS for handling cross-origin requests

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load the trained model
model = load_model('binary_classification_model.h5')

# Define image dimensions expected by the model
IMG_WIDTH, IMG_HEIGHT = 150, 150  # Adjust as per your model's input shape


@app.route('/get-ip', methods=['GET'])
def get_ip():
    return jsonify({'ip': request.host}), 200
# Connect to SQLite database and create users table if it doesn't exist
def init_sqlite_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

# Initialize the database
init_sqlite_db()

# Route for handling image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Process the uploaded image without saving it
        img = Image.open(BytesIO(file.read()))

        # Convert image to RGB mode if it's in RGBA or other mode
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Preprocess the image for the model
        image = img.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize to match model input shape
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make the prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Map prediction to the respective class
        class_labels = {0: 'Organic', 1: 'Recyclable'}  # Adjust according to your model
        result = class_labels[predicted_class]

        return jsonify({'prediction': result})

# Route for user registration
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not all([name, email, password]):
        return jsonify({'error': 'Please fill out all fields'}), 400

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                       (name, email, hashed_password))
        conn.commit()
        conn.close()
        return jsonify({'message': 'User registered successfully'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already exists'}), 400
# Route for user login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not all([email, password]):
        return jsonify({'error': 'Please fill out all fields'}), 400

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user[0], password):
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'error': 'Invalid email or password'}), 401

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    
