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

url = 'https://virtual-ai-stylist-backend.onrender.com/upload'
headers = {
    "Accept": 'application/json',
}

@app.route('/upload-ai-image', methods = ['POST']) 
def post(): 
    data = request.get_data().decode('ascii')
    input_path = 'input\input.jpg'
    url = urlopen(data)
    per_array = asarray(bytearray(url.read()), dtype = 'uint8')
    person = imdecode(per_array, IMREAD_COLOR)
    imwrite(input_path, person)
    overlay()

    files = {
    'image' : open(output_path, 'rb')
    }
    link = requests.post(url, headers=headers, files=files).json()['url']
    print(link)
    return 'https://virtual-ai-stylist-backend.onrender.com' + link

if __name__ == "__main__":     
    app.run(debug=True, host='0.0.0.0')