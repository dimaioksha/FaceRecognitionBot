import numpy as np
import imutils
import cv2
import dlib
from typing import Tuple, List, Union

# magical consts for model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
GENDER = ['Male', 'Female']
REQUIRED_WIDTH = 300


class FullModel:
    """
    Main model class. Before usage download required weights for it.
    """
    def __init__(self):
        """
        Setting weights to the models
        """
        self.net = cv2.dnn.readNetFromCaffe("../detectionbot/model/deploy.prototxt", "../detectionbot/model/res10_300x300_ssd_iter_140000.caffemodel")
        self.age_net = cv2.dnn.readNetFromCaffe(
            "../detectionbot/model/age_gender_models/deploy_age.prototxt",
            "../detectionbot/model/age_gender_models/age_net.caffemodel")
        self.gender_net = cv2.dnn.readNetFromCaffe(
            "../detectionbot/model/age_gender_models/deploy_gender.prototxt",
            "../detectionbot/model/age_gender_models/gender_net.caffemodel")

    @staticmethod
    def calculate_shapes(frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Static method which calculates required_height and factor for models.
        :param frame: np.ndarray
        :return: Tuple[np.ndarray, int]
        """
        src_height, src_width = frame.shape[:2]
        factor = src_width / REQUIRED_WIDTH
        required_height = src_height / factor
        return required_height, factor

    @staticmethod
    def preprocess_image(frame: np.ndarray) -> np.ndarray:
        """
        Static transpose and preprocess image before in model usage
        :param frame: np.ndarray
        :return: np.ndarray
        """
        det_input = frame.transpose(2, 0, 1)
        det_input = det_input.reshape(1, *det_input.shape)
        return det_input

    def _predict(self, frame: np.ndarray, rects: list, *consts: Tuple)\
            -> Tuple[list, List[Union[str, List[str]]], List[Union[str, List[str]]]]:
        """
        Protected _predict function. Hides all inside predictions in one function.
        :param frame: np.ndarray
        :param rects: list
        :param consts: Tuple
        :return: Tuple[list, List[Union[str, List[str]]], List[Union[str, List[str]]]]
        """
        def _prepare_coords() -> Tuple[int, int, int, int]:
            """
            Preprocess coords of rectangle. Hidden in _predict() as like in wrapper
            :return: Tuple[int, int, int, int]
            """
            x1, y1, x2, y2 = rect
            x1, y1, x2, y2 = int(x1 * REQUIRED_WIDTH), int(y1 * required_height), \
                             int(x2 * REQUIRED_WIDTH), int(y2 * required_height)
            # extension
            a = 0.1
            aw = a * (x2 - x1)
            ah = a * (y2 - y1)
            x1, x2, y1, y2 = np.array([x1 - aw, x2 + aw, y1 - ah, y2 + ah]).astype(int)
            return x1, y1, x2, y2

        def _get_predictions(flag: str) -> str:
            """
            Uses self attributes to make predictions depending on flag
            :param flag: str
            :return: str
            """
            if flag not in ('gender', 'age'):
                raise ValueError("Not existing flag for prediction")
            if flag == 'gender':
                self.gender_net.setInput(blob2)
                return GENDER[self.gender_net.forward()[0].argmax()]
            elif flag == "age":
                self.age_net.setInput(blob2)
                return AGE[self.age_net.forward()[0].argmax()]

        required_height, factor = consts
        new_rects, genders, ages = [], [], []
        # use cycle because of many faces which could be on the picture
        for rect in rects:
            if rect[0] < 0.9:
                continue
            rect = rect[1:]
            x1, y1, x2, y2 = _prepare_coords()
            new_rects.append(np.array([x1, y1, x2, y2]) * factor)
            blob2 = cv2.dnn.blobFromImage(frame[y1:y2, x1:x2].copy(), 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genders.append(_get_predictions(flag='gender'))
            ages.append(_get_predictions(flag='age'))

        return new_rects, genders, ages

    def predict(self, frame: np.ndarray) -> Tuple[list, List[Union[str, List[str]]], List[Union[str, List[str]]]]:
        """
        Function to predict rectangle, age and sex
        :param frame: np.ndarray
        :return: Tuple[list, List[Union[str, List[str]]], List[Union[str, List[str]]]]
        """
        height, factor = FullModel.calculate_shapes(frame)
        frame = imutils.resize(frame, width=REQUIRED_WIDTH)
        self.net.setInput(FullModel.preprocess_image(frame))
        rects = self.net.forward()
        rects = rects.reshape(*rects.shape[2:])[:, 2:]
        return self._predict(frame,\
                             rects,\
                             height, factor)

def transform(frame, rects, genders, ages) -> np.ndarray:
    """
    Function to put on picture predicted text and rectangles
    :param frame: np.ndarray
    :param rects: list
    :param genders: List[Union[str, List[str]]]
    :param ages: List[Union[str, List[str]]]
    :return: np.ndarray
    """
    frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for rect, gender, age in zip(rects, genders, ages):
        # coordinates of rectangle
        x1, y1, x2, y2 = rect.astype(int)
        # text
        overlay_text = "{}, {}".format(gender, age)
        cv2.putText(frame, overlay_text, (x1, y1), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame
