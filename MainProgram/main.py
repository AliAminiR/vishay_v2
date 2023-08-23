import json
from datetime import datetime
import os
import argparse

from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import torch
from PIL import Image
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app)

camera = None
chip_checker = None
path_to_json_folder = None
path_to_img_folder = None
path_to_model = None


# path_to_json_folder = r"C:\Users\A750290\Projects\Vishay\python\MainProgram\output"
# path_to_img_folder = r"C:\Users\A750290\Projects\Vishay\python\MainProgram\Saved_images"
# path_to_model = r"C:\Users\A750290\Projects\Vishay\python\Classification\NetModel_first best.pth"

class VishayModel:
    classes = {"0": "NiO", "1": "iO"}
    model = None
    _path_to_model = None

    def __init__(self, model_path):
        self._path_to_model = model_path
        model = resnet18(weights="ResNet18_Weights.DEFAULT")
        model.fc = torch.nn.Linear(512, 2)
        model.load_state_dict(torch.load(self._path_to_model))
        model.eval()

        if device == "cuda":
            model = model.cuda()

        self.model = model

    def get_prediction(self, input_tensor):
        outputs = self.model.forward(input_tensor)
        _, y_hat = outputs.max(1)
        prediction = y_hat.item()
        return prediction

    @staticmethod
    def transform_image(image):
        input_transforms = transforms.Compose([transforms.Resize((512, 512)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                               ])
        timg = input_transforms(image)
        timg.unsqueeze_(0)
        return timg

    def render_prediction(self, prediction_idx):
        stridx = str(prediction_idx)
        class_name = 'Unknown'
        if self.classes is not None:
            if stridx in self.classes is not None:
                class_name = self.classes[stridx]
        return prediction_idx, class_name


class Camera:  # constantly reads frames from a specified webcam.
    def __init__(self, cam_index):
        self.cam_index = cam_index
        self.cam = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cam.isOpened():
            raise ValueError(f"Camera {cam_index} not found or could not be opened.")

    def capture_frame(self):  # captures a frame from the camera. If the frame was read successfully, it returns it.
        ret, frame = self.cam.read()
        if ret:
            return frame
        else:
            print(f"Error reading frame from camera {self.cam_index}")
            return None


@app.route('/', methods=['GET'])
def root():
    return jsonify(
        {'msg': 'Try a GET request to the /predict/<x>/<y>/<pos> endpoint'})


@app.route('/predict/<x>/<y>/<pos>', methods=['GET'])
def infer_image(x, y, pos):
    now = datetime.now()
    img = camera.capture_frame()
    if img is not None:
        image_name = f"{now.strftime('%Y%m%d-%H%M%S')}_{x}_{y}_{pos}.png"
        image_path = os.path.join(path_to_img_folder, image_name)
        try:
            cv2.imwrite(image_path, img)
        except Exception as e:
            print('Error happened while saving the taken image:', e)
            return jsonify({'status': f'Error -->  {e}'})

        class_name = "Unknown"
        class_id = "Unknown"
        infer_status = "Unknown"

        try:
            image = Image.open(image_path)
            input_tensor = chip_checker.transform_image(image)
            prediction_idx = chip_checker.get_prediction(input_tensor)
            class_id, class_name = chip_checker.render_prediction(prediction_idx)
            infer_status = "Done"

        except Exception as e:
            print('Error happened while inferring image:', e)
            infer_status = "Error"

        finally:
            output_json = {'class_id': class_id,
                           'class_name': class_name,
                           'image_name': image_name,
                           'infer_status': infer_status,
                           "timestamp": now.strftime('%Y%m%d-%H%M%S'),
                           'x': str(x),
                           'y': str(y),
                           'POS': str(pos)}

        try:
            with open(os.path.join(path_to_json_folder, f"{now.strftime('%Y%m%d-%H%M%S')}_{x}_{y}_{pos}.json"),
                      "w") as outfile:
                json.dump(output_json, outfile)
            return jsonify({'status': f'{infer_status}'})
        except Exception as e:
            print('Error happened while saving Json file:', e)
            return jsonify({'status': f'Error -->  {e}'})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_target_dir", type=str, required=True,
                    help="path to a image target directory for saving the taken images")
    ap.add_argument("-j", "--json_target_dir", type=str, required=True,
                    help="path to a json target directory for saving the predicted results")
    ap.add_argument("-m", "--model", type=str, required=True,
                    help="path to the AI-model")
    args = vars(ap.parse_args())

    assert args["image_target_dir"] is not None, "define the image target directory"
    path_to_img_folder = args["image_target_dir"]

    assert args["json_target_dir"] is not None, "define the json target directory"
    path_to_json_folder = args["json_target_dir"]

    assert args["model"] is not None, "define the model path"
    path_to_model = args["model"]

    camera = Camera(0)
    chip_checker = VishayModel(path_to_model)
    # infer_image("1","2","snaf")
    app.run(debug=True, host='0.0.0.0')
