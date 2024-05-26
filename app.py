import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Cargar el modelo entrenado
model = tf.keras.models.load_model('model_lenet.h5')

def preprocess_image(image):
    image = image.convert('L').resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_number = np.argmax(prediction)
    return jsonify({'number': int(predicted_number)})

if __name__ == '__main__':
    app.run(debug=True)
