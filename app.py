from flask import Flask
from flask_restful import Api, Resource
from overlaying import overlay
from json import dumps, loads
from cv2 import imwrite
from os import getcwd
app = Flask(__name__)
api = Api(app)

class API(Resource):
    def post(image_URL):

        output_path = 'output\output.jpg'
        imwrite(output_path, overlay(image_URL))

        return loads(dumps({
            'name': 'AIPic',
            'type': 'image/jpg',
            'uri': getcwd() + '\\' + output_path
        }))


api.add_resource(API, '/get-cut-clothes')

if __name__ == '__main__':
    app.run(debug=True)