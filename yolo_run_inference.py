#For yolo inference
import glob
import numpy as np
import onnxruntime
from yolo_onnx_preprocessing_utils import non_max_suppression, _convert_to_rcnn_output

#--processing--
def get_predictions_from_ONNX(onnx_session, img_data):
    """perform predictions with ONNX Runtime

        :param onnx_session: onnx model session
        :type onnx_session: class InferenceSession
        :param img_data: pre-processed numpy image
        :type img_data: ndarray with shape 1xCxHxW
        :return: boxes, labels , scores
        :rtype: list
        """
    sess_input = onnx_session.get_inputs()
    sess_output = onnx_session.get_outputs()
    # predict with ONNX Runtime
    output_names = [output.name for output in sess_output]
    pred = onnx_session.run(
    output_names=output_names, input_feed={sess_input[0].name: img_data}
    )
    return pred[0]

#--post-processing--
def _get_box_dims(image_shape, box):
    box_keys = ['topX', 'topY', 'bottomX', 'bottomY']
    height, width = image_shape[0], image_shape[1]

    box_dims = dict(zip(box_keys, [coordinate.item() for coordinate in box]))

    box_dims['topX'] = box_dims['topX'] * 1.0 / width
    box_dims['bottomX'] = box_dims['bottomX'] * 1.0 / width
    box_dims['topY'] = box_dims['topY'] * 1.0 / height
    box_dims['bottomY'] = box_dims['bottomY'] * 1.0 / height

    return box_dims

def _get_prediction(label, image_shape, classes):   
    boxes = np.array(label["boxes"])
    labels = np.array(label["labels"])
    labels = [label[0] for label in labels]
    scores = np.array(label["scores"])
    scores = [score[0] for score in scores]

    bounding_boxes = []
    for box, label_index, score in zip(boxes, labels, scores):
        box_dims = _get_box_dims(image_shape, box)

        box_record = {'box': box_dims,
                    'label': classes[label_index],
                    'score': score.item()}

        bounding_boxes.append(box_record)

    return bounding_boxes
