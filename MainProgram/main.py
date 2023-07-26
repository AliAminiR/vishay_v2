from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


@app.route('/predict/<x>/<y>/<pos>', methods=['GET'])
def infer_image(x, y, pos):
    # TODO: get image from Camera
    # TODO: Prepare the image for inference
    # TODO: Make a prediction
    # TODO: Save prediction in JSON
    # TODO: Succefully save the json on Shared Folder
    return "X= " + str(x) + " Y= " + str(y) + " POS= " + str(pos)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')