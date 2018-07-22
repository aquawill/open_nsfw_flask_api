#!/usr/bin/env python
import ast
import os
import random
import ssl
from urllib.request import Request, urlopen

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask
from flask_restful import reqparse, Api, Resource
from tensorflow.python.ops import variable_scope as vs

from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader
from model import OpenNsfwModel, InputType

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"
context = ssl._create_unverified_context()


def classifier(img, url, reuse=False):
    if reuse:
        vs.get_variable_scope().reuse_variables()
    image_loader = 'yahoo'
    input_file = img
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
            # fn_load_image = img

        sess.run(tf.global_variables_initializer())
        image = fn_load_image(input_file)
        predictions = sess.run(model.predictions, feed_dict={model.input: image})
        sess.close()
    return {'url': url, 'sfw': str(predictions[0][0]), 'nsfw': str(predictions[0][1])}


app = Flask(__name__)


def dl_img(img_url):
    req = Request(img_url)
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36")
    img = urlopen(req, context=context)
    arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
    return arr

def url_processor(url):
    img_arr = dl_img(url)
    im = cv2.imdecode(img_arr, -1)
    file_name = random.randint(0, 9999999999999999)
    cv2.imwrite('{}.jpg'.format(str(file_name)), im)
    result = classifier('{}.jpg'.format(str(file_name)), url)
    os.remove('{}.jpg'.format(str(file_name)))
    return result


class nsfw_image_api(Resource):
    def post(self):
        results = []
        args = parser.parse_args()
        print(args)
        if args.get('urls'):
            urls = ast.literal_eval(args.get('urls'))
            for _ in range(len(urls)):
                url = urls[_]
                img_arr = dl_img(url)
                im = cv2.imdecode(img_arr, -1)
                file_name = random.randint(0, 9999999999999999)
                cv2.imwrite('{}.jpg'.format(str(file_name)), im)
                result = classifier('{}.jpg'.format(str(file_name)), url, reuse=(_ != 0))
                os.remove('{}.jpg'.format(str(file_name)))
                results.append(result)
        elif args.get('url'):
            url = args.get('url')
            #url_processor(url)
            img_arr = dl_img(url)
            im = cv2.imdecode(img_arr, -1)
            file_name = random.randint(0, 9999999999999999)
            cv2.imwrite('{}.jpg'.format(str(file_name)), im)
            result = classifier('{}.jpg'.format(str(file_name)), url)
            os.remove('{}.jpg'.format(str(file_name)))
            results.append(result)
        return results


app = Flask(__name__)
api = Api(app)
api.add_resource(nsfw_image_api, '/', endpoint='url')
parser = reqparse.RequestParser()
parser.add_argument('url')
parser.add_argument('urls')

if __name__ == "__main__":
    app.run(port=5001)
    # #http://n.sinaimg.cn/finance/transform/20161206/BDjF-fxyiayr9304919.jpg
