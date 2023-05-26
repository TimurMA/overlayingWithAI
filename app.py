from flask import Flask, request
import requests
from flask_restful import Api
from overlaying import overlay
from cv2 import imwrite, imdecode, IMREAD_COLOR
from urllib.request import urlopen
from numpy import asarray
app = Flask(__name__)
api = Api(app)

output_path = 'output/output.jpg'
@app.route('/upload-ai-image', methods = ['POST']) 
def post(): 
    # data = request.get_data().decode('ascii')
    # input_path = 'input\input.jpg'
    # url = urlopen(data)
    # per_array = asarray(bytearray(url.read()), dtype = 'uint8')
    # person = imdecode(per_array, IMREAD_COLOR)
    # imwrite(input_path, person)
    # overlay()

    url = 'https://virtual-ai-stylist-backend.onrender.com/upload'
    files = {
        'image' : open(output_path, 'rb')
    }
    headers = {
        "Accept": 'application/json',
        # 'Content-Type': 'multipart/form-data',
      }
    data = requests.post(url, headers=headers, files=files)
    link = 'https://virtual-ai-stylist-backend.onrender.com' + data.json()['url']
    return link

    


if __name__ == "__main__":     app.run(debug=True, host='0.0.0.0')