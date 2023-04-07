# DAC Systems Design Contest 2023 Team ~BFGPU


Branches:
- `inference`: The code to run inference is in this branch. We have set up an API to allow the testing of models (given in .onnx) right on the device. 
- `vit` : The code for the ViT based model. 
- `yolov8`: the code for the yolov8 based model.

We use poetry as our dependency manager. Please use `poetry install` to get all the required dependencies. 