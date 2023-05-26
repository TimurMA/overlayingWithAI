from flask import Flask, request, send_file
from flask_restful import Api
from overlaying import overlay
from cv2 import imwrite, imdecode, IMREAD_COLOR
from urllib.request import urlopen
from numpy import asarray
from PIL import Image
import io
app = Flask(__name__)
api = Api(app)

output_path = 'output/output.jpg'
@app.route('/output.jpg', methods = ['POST']) 
def post(): 
    data = request.get_data().decode('ascii')
    input_path = 'input\input.jpg'
    url = urlopen(data)
    per_array = asarray(bytearray(url.read()), dtype = 'uint8')
    person = imdecode(per_array, IMREAD_COLOR)
    imwrite(input_path, person)
    overlay()

    file_object = io.BytesIO()
    img = Image.open(output_path)
    img.save(file_object, 'JPG')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/JPG')


if __name__ == "__main__": 
    app.run(debug=True, host='0.0.0.0')