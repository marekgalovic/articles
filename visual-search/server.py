import os
import json
from PIL import Image
from io import BytesIO

from flask import Flask, jsonify, request, render_template, send_file

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from annoy import AnnoyIndex

_model = None
_index = None

def get_model():
    global _model
    if _model is None:
        _model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.image.convert_image_dtype(x, tf.float32)),
            hub.KerasLayer("https://tfhub.dev/tensorflow/resnet_50/feature_vector/1", trainable=False)  
        ])

    return _model


def get_index():
    global _index
    if _index is None:
        idx = AnnoyIndex(2048, 'euclidean')
        idx.load(os.path.join(os.getenv('INDEX_PATH'), 'index.ann'))
        metadata = json.load(open(os.path.join(os.getenv('INDEX_PATH'), 'index_metadata.json'), 'r'))
        metadata = {int(k): v for (k, v) in metadata.items()}
        _index = (idx, metadata)

    return _index


def open_img(data):
    img = Image.open(BytesIO(data))
    return img.resize((224, 224))


def get_feature_vector(img):
    model = get_model()

    return model(np.asarray(img).reshape((-1, 224, 224, 3)))[0]


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/images/<fname>', methods=['GET'])
def images(fname):
    return send_file(os.path.join('images', fname))


@app.route('/search', methods=['POST'])
def search():
    if request.content_type != 'image/jpeg':
        return jsonify({
            'error': 'Image must be a jpeg'
        }), 400

    img = open_img(request.get_data())
    vec = get_feature_vector(img)

    index, index_metadata = get_index()
    ids = index.get_nns_by_vector(vec, 10)
    return jsonify({
        'items': [
            {'id': id, 'metadata': index_metadata.get(id, None)}
            for id in ids
        ]
    })
