# license-plate-reader
A quick license plate reader I hacked together in a week.  
Uses YOLO to detect objects and the license plates of vehicles (a pretrained model I found online), then extracts the digits out of the plate and passes to tesseract.  
In my experience tessaract wasn't relaible enough, so I would like to train a NN model to recognize the digits in the future.  

sources:  
[lisence plate model](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8)  
[digit extraction](https://github.com/theAIGuysCode/yolov4-custom-functions)  
[training a digit model](https://github.com/ShabbirMK/Handwritten-Number-Recognition-using-Deep-Learning-PyTorch-and-GUI)  
