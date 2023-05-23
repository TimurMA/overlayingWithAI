from flask import Flask, request
from flask_restful import Api, Resource
from overlaying import overlay
from json import dumps, loads
from cv2 import imwrite
from os import getcwd
app = Flask(__name__)
api = Api(app)

@app.route('/get-cut-clothes', methods = ['POST']) 
def post(): 
    data = request.get_data().decode('ascii')
    output_path = 'output\output.jpg'
    imwrite(output_path, overlay(data))

    return loads(dumps({
            'name': 'AIPic',
            'type': 'image/jpg',
            'uri': getcwd() + '\\' + output_path
        }))

if __name__ == "__main__": 
    app.run(debug=True)