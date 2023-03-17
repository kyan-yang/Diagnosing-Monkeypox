import os

from flask import Flask
from flask import request
from flask import render_template

import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.io import imread
from PIL import Image


app = Flask(__name__)
UPLOAD_FOLDER = "https://github.com/kyan-yang/Diagnosing-Monkeypox/tree/main/static"
MODEL = None

def read_image(filepath):
    arr = np.zeros((1,224,224,3))
    arr[0,:,:,:] = imread(filepath)
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
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            img = Image.open(image_location)
            width, height = img.size
            if width != 224 and height != 224:
                img2 = make_square(img).resize((224,224), Image.LANCZOS)
                img2.save(image_location)
            pred = predict(image_location, MODEL)
            print(pred)
            diagnosis = "Negative"
            if bool(pred[0,0]):
                diagnosis = "Positive"
            return render_template("index.html", prediction=diagnosis, image_loc=image_file.filename)
    return render_template("index.html", prediction=0, image_file=None)

if __name__ == "__main__":
    MODEL = keras.models.load_model('https://github.com/kyan-yang/Diagnosing-Monkeypox/tree/main/model_8752.h5')
    app.run(port=12000, debug=True)
