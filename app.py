from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
from flask_cors import CORS
import base64
import os

app = Flask(__name__)

CORS(app)

@app.route('/')
def index():
    return jsonify({'message': 'Hello world'})


@app.route('/image', methods=['POST'])
def image():
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'})

    image_data = request.files['image']
    image_array = np.frombuffer(image_data.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'No se pudo decodificar la imagen'})

    result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
    dominantEmotion = result[0]['dominant_emotion']
    emotions = result[0]['emotion']

    response = {'dominantEmotion': dominantEmotion, "emotion": emotions}
    return jsonify(response)

""" # Image in base64
@app.route('/base64', methods=['POST'])
def base64():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No se envió ninguna imagen'})

    image_data = data['image']
    image_data = base64.b64decode(image_data)

    # Save the decoded image to the "uploads" directory
    image_path = os.path.join("uploads", "decoded_image.jpg")
    with open(image_path, "wb") as f:
        f.write(image_data)

    return jsonify({'message': 'Image saved successfully'})
    
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'No se pudo decodificar la imagen'})

    result = DeepFace.analyze(image, actions=['emotion'])
    dominantEmotion = result['emotion']['dominant']
    emotions = result['emotion']

    response = {'dominantEmotion': dominantEmotion, "emotion": emotions}
    return jsonify(response) """

if __name__ == '__main__':
    app.run(debug=True)