from pathlib import Path
import wandb
from fastapi import FastAPI, UploadFile

#For yolo inference
import glob
import numpy as np
import onnxruntime
from yolo_onnx_preprocessing_utils import non_max_suppression, _convert_to_rcnn_output
from yolo_run_inference import get_predictions_from_ONNX, _get_box_dims, _get_prediction
app = FastAPI()


@app.post("/yolo")
def yolo_fmt(
    model: UploadFile,
    run_name: str,
    img_size: int = 640,
):
    # model_file = Path("model.onnx")
    # model_file.write_bytes(model.file.read())


    #--pre-processing--
    start_time = time.time()
    onnx_model_path = "best.onnx" # replace with file path
    try:
        session = onnxruntime.InferenceSession(onnx_model_path)
        print("ONNX model loaded...")
    except Exception as e:
        print("Error loading ONNX file: ", str(e))

    batch_size, channel, height_onnx, width_onnx = session.get_inputs()[0].shape

    print(batch_size, channel, height_onnx, width_onnx)

    # use height and width based on the generated model
    test_images_path = (
        "automl_models_od_yolo/test_images_dir/*"  # replace with path to images
    )
    image_files = glob.glob(test_images_path)
    img_processed_list = []
    pad_list = []
    for i in range(batch_size):
        img_processed, pad = preprocess(image_files[i])
        img_processed_list.append(img_processed)
        pad_list.append(pad)

    if len(img_processed_list) > 1:
        img_data = np.concatenate(img_processed_list)
    elif len(img_processed_list) == 1:
        img_data = img_processed_list[0]
    else:
        img_data = None

    assert batch_size == img_data.shape[0]

    #--processing--
    result = get_predictions_from_ONNX(session, img_data)

    #--post-processing--
    total_time = (time.time() - start_time())  / 1000 #time in seconds
    result_final = non_max_suppression(
    torch.from_numpy(result), conf_thres=0.1, iou_thres=0.5
    )

    bounding_boxes_batch = []
    for result_i, pad in zip(result_final, pad_list):
        label, image_shape = _convert_to_rcnn_output(result_i, height_onnx, width_onnx, pad)
        bounding_boxes_batch.append(_get_prediction(label, image_shape, classes))
    print(json.dumps(bounding_boxes_batch, indent=1))
    
    #Still need to implement power consumption and put results onto wandb

@app.post("/coco")
def coco_fmt(model: UploadFile, img_size):
    pass

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
