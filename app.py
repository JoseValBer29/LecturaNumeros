import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
import sys
from PIL import Image
import logging

app = Flask(__name__)

# Cargar el modelo entrenado
model = tf.keras.models.load_model('model_lenet.h5')

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

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
    logging.debug(f'image_data = {image_data}')
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    logging.debug('Tamaño de la imagen: %s', image.size)
    processed_image = preprocess_image(image)
    logging.debug('Forma de la imagen procesada: %s', processed_image.shape)
    prediction = model.predict(processed_image)
    predicted_number = np.argmax(prediction)
    logging.debug('Predicción: %s', predicted_number)
    return jsonify({'number': int(predicted_number)})

if __name__ == '__main__':
    app.run(debug=True)
