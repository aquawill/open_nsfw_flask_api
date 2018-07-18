#!/usr/bin/env python
import io
from urllib.request import Request, urlopen
import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import random

from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader
from model import OpenNsfwModel, InputType


IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"

def classifier(img):
    image_loader = 'yahoo'
    input_file = 'test.jpg'
    input_type = 'tensor'
    model_weights = 'data/open_nsfw-weights.npy'

    model = OpenNsfwModel()

    with tf.Session() as sess:

        input_type = InputType[input_type.upper()]
        model.build(weights_path=model_weights, input_type=input_type)

        fn_load_image = None

        if input_type == InputType.TENSOR:
            if image_loader == IMAGE_LOADER_TENSORFLOW:
                fn_load_image = create_tensorflow_image_loader(sess)
            else:
                fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            import base64
            fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])
            #fn_load_image = img

        sess.run(tf.global_variables_initializer())
        image = fn_load_image(input_file)
        predictions = sess.run(model.predictions, feed_dict={model.input: image})
    return {'sfw': str(predictions[0][0]), 'nsfw': str(predictions[0][1])}

app = Flask(__name__)

def dl_img(img_url):
    req = Request(img_url)
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36")
    img = urlopen(req)
    arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
    return arr

class nsfw_image_api(Resource):
    def post(self):
        args = parser.parse_args()
        url = args['url']
        img_arr = dl_img(url)
        im = cv2.imdecode(img_arr, -1)
        print(io.BytesIO(cv2.imencode('.jpg', im)[1].tostring()))
        file_name = random.randint(0, 9999999999999999)
        cv2.imwrite('{}.jpg'.format(str(file_name)), im)
        result = classifier('{}.jpg'.format(str(file_name)))
        os.remove('{}.jpg'.format(str(file_name)))
        return result


app = Flask(__name__)
api = Api(app)
api.add_resource(nsfw_image_api, '/', endpoint='url')
parser = reqparse.RequestParser()
parser.add_argument('url')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
    # #http://n.sinaimg.cn/finance/transform/20161206/BDjF-fxyiayr9304919.jpg