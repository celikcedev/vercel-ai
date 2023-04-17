import os
from flask import Flask, render_template, request
import torch
import numpy as np
import cv2
from PIL import Image
from yolov5.utils.general import non_max_suppression
# from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from yolov5.utils.torch_utils import time_sync
from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load
#from yolov5.utils.datasets import letterbox
#from yolov5.utils.plots import colors, plot_one_box
from yolov5.utils.plots import colors
from yolov5.utils.augmentations import letterbox
from yolov5.utils import *


app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)  # .autoshape(), yüklenen modeli autoshape ile şekillendiriyoruz
model = model.to(device)

def detect(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # RGB formatındaki görüntüyü BGR formatına çeviriyoruz
    results = model(img, size=640)
    #results.show()
    results.save(save_dir='static\images', exist_ok=True)
    results.show('static\images\image0.jpg') 
    return results.pandas().xyxy[0].to_json(orient="records")
    
    # results.pandas().xyxy[0]  # im predictions (pandas)
    #      xmin    ymin    xmax   ymax  confidence  class    name
    # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
    
    #results.pandas().xyxy[0].value_counts('name')  # class counts (pandas)
    # person    2
    # tie       1

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        image = file.read()
        npimg = np.fromstring(image, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        detection_result = detect(img)
        return detection_result
    else:
        return render_template('index.html')

if __name__ == '__main__':
    # app.run(debug=True)
    app.run()
 