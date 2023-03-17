import os

from flask import Flask
from flask import request
from flask import render_template

import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.io import imread
from PIL import Image

import tempfile
import shutil
import base64
import io

app = Flask(__name__)
MODEL = None

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
            pred = predict(image_location, MODEL)
            print(pred)
            diagnosis = "Negative"
            if bool(pred[0,0]):
                diagnosis = "Positive"
            img.close()
            im.close()
            shutil.rmtree(dir)
            return render_template("index.html", prediction=diagnosis, image_data=encoded_img_data.decode('utf-8'))
    return render_template("index.html", prediction=0, image_file=None)

if __name__ == "__main__":
    MODEL = keras.models.load_model('/opt/render/project/src/.venv/lib/python3.7/site-packages/flask/model_8752.h5')
    app.run(port=12000, debug=True)
