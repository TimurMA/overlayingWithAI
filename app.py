from flask import Flask
from flask_restful import Api, Resource
from overlaying import overlay
from json import dumps, loads
from cv2 import imwrite
app = Flask(__name__)
api = Api(app)

class API(Resource):
    def post(image_URL):

        output_path = 'D:\Python37\Overlaying\output\output.jpg'
        imwrite(output_path, overlay(image_URL))

        return loads(dumps({
            'name': 'AIPic',
            'type': 'image/jpg',
            'uri': output_path
        }))


api.add_resource(API, '/get-cut-clothes')

if __name__ == '__main__':
    app.run(debug=True)