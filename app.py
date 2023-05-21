from flask import Flask
from flask_restful import Api, Resource
from overlaying import overlay
from cv2 import imwrite
from json import dumps, loads

app = Flask(__name__)
api = Api(app)

class API(Resource):
    def post(image):
        output_path = 'D:\Python37\Overlaying\VAISoutput/output.jpg'
        imwrite(output_path, overlay(image))

        return loads(dumps({
            'name': 'AIPic',
            'type': 'image/jpg',
            'uri': output_path
        }))


api.add_resource(API, '/get-cut-clothes')

if __name__ == '__main__':
    app.run(debug=True)