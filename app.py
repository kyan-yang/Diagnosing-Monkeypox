import os

from flask import Flask
from flask import request
from flask import render_template

import tempfile
import shutil
import base64
import io

import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.regularizers import l2

from skimage.io import imread
from PIL import Image
app = Flask(__name__)

def sigmoid(x):
  return 1/(1 + math.exp(-x))

def import_model(filepath):
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224,224, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.12),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.12),
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(l=0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.12),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(l=0.007)),
        Dropout(0.5),
        Dense(1),
    ])
    model.load_weights(filepath)
    return model

def read_image(image_path):
    arr = np.zeros((1,224,224,3))
    arr[0,:,:,:] = imread(image_path)
    return arr

def predict(image_path, model):
    X_test = read_image(image_path)
    y = model.predict(X_test)
    return y

def make_square(im, min_size=224, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

@app.route('/', methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            dir = tempfile.mkdtemp()
            image_location = os.path.join(dir, image_file.filename)
            image_file.save(image_location)
            # resize
            img = Image.open(image_location)
            width, height = img.size
            if width != 224 and height != 224:
                img2 = make_square(img).resize((224,224), Image.LANCZOS)
                img2.save(image_location)
                img2.close()
            # save
            im = Image.open(image_location)
            data = io.BytesIO()
            im.save(data, "JPEG")
            encoded_img_data = base64.b64encode(data.getvalue())
            MODEL = import_model('model_8781.hdf5')
            MODEL.compile(optimizer=keras.optimizers.Adam(epsilon=0.01), loss='binary_crossentropy', metrics=['binary_accuracy'])
            pred = predict(image_location, MODEL)
            img.close()
            im.close()
            shutil.rmtree(dir)
            conf = 0
            pred = pred/500
            pred = sigmoid(pred)
            diagnosis = "n/a"
            if (pred > 0.2):
                diagnosis = "Positive"
                conf = 0.625*pred + 0.375
            else:
                diagnosis = "Negative"
                conf = 1 - 2.5*pred
            conf = int(conf*100)
            return render_template("index.html", prediction=diagnosis, confidence=conf, image_data=encoded_img_data.decode('utf-8'))
    return render_template("index.html", prediction=0, image_file=None)

if __name__ == "__main__":
    app.run(debug=True)