import cv2.dnn
import numpy as np


class yolo_13h:
    def __init__(self, fenlei_model_path, jishu_model_path):
        # Load the ONNX model
        self.model_fenlei: cv2.dnn.Net = cv2.dnn.readNetFromONNX(fenlei_model_path)
        self.model_fenlei.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model_fenlei.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model_jishu: cv2.dnn.Net = cv2.dnn.readNetFromONNX(jishu_model_path)
        self.model_jishu.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model_jishu.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.class_indict_fenlei = {0: 'feike', 1: 'feilian', 2: 'jinpu'}
        self.class_indict_jishu = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}

    def fenlei(self, image_path):
        # Read the input image
        original_image: np.ndarray = cv2.imread(image_path)
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, swapRB=True)
        self.model_fenlei.setInput(blob)

        # Perform inference
        outputs = self.model_fenlei.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        return self.class_indict_fenlei[np.argmax(outputs, axis=2)[0][0]], np.amax(outputs, axis=2)

    def jishu(self, image_path):
        # Read the input image
        original_image: np.ndarray = cv2.imread(image_path)
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, swapRB=True)
        self.model_jishu.setInput(blob)

        # Perform inference
        outputs = self.model_jishu.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        return self.class_indict_jishu[np.argmax(outputs, axis=2)[0][0]], np.amax(outputs, axis=2)



if __name__ == "__main__":
    image_path = '/home/jjgong/colony_count/test/肺链_13h/C00000047432_6_1145.jpg'
    yolofenlei = yolo_13h("models/分类模型_三种菌_13h.onnx", "models/计数模型_三种菌.onnx")
    print(yolofenlei.fenlei(image_path))
    print(yolofenlei.jishu(image_path))