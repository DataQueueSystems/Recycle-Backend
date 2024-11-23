import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)



# Load the pre-trained model once at the beginning
MODEL_PATH = 'binary_classification_model.h5'
model = load_model(MODEL_PATH)



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


# Function to preprocess and predict a single image
def preprocess_and_predict(image_path, target_size=(150, 150)):
    """
    Preprocess the image and predict its class.

    Parameters:
        image_path (str): Path to the image file for prediction.
        target_size (tuple): Target size to resize the image.

    Returns:
        dict: Prediction results containing class and probability.
    """
    # Load and preprocess the image
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)  # Convert image to numpy array
    image_array = preprocess_input(image_array)  # Preprocess the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(image_array)[0][0]
    predicted_class = 'Recyclable' if prediction > 0.5 else 'Organic'
    if predicted_class == 'Recyclable':
        tips = [
        "Sort the recyclable materials into categories (e.g., paper, plastic, glass, etc.).",
        "Clean the items before recycling to avoid contamination.",
        "Check local recycling rules to ensure the items are accepted in your area."
        ]
    else:
        tips = [
        "Compost organic waste to create nutrient-rich soil for gardening.",
        "Separate food scraps from general waste and compost them properly.",
        "Use a compost bin or a composting pile in your garden for better waste management."
        ]
    return {"class": predicted_class, "probability": float(prediction),"tips": tips}

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Create an `index.html` file in the `templates` folder

# Route for handling image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        # Save the uploaded file temporarily
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)  # Ensure uploads directory exists
        file.save(file_path)

        # Perform prediction
        result = preprocess_and_predict(file_path)

        # Remove the temporary file after prediction
        os.remove(file_path)

        return jsonify(result)




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





if __name__ == '__main__':
    # Ensure the server is running in debug mode
    app.run(debug=True,host='0.0.0.0', port=5000)
