from torchvision import transforms
import os
import torch
import numpy as np
from PIL import Image
from three_DConv import ResNet_3d
from three_DConv import Bottleneck
from filename_sorted import sort_by_numeric_value

import cv2.dnn


class fenlei_3D:
    def __init__(self, model_path, jishu_model_path):
        self.data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ])
        # load model weights
        assert os.path.exists(model_path), "file: '{}' dose not exist.".format(model_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # create model
        self.model = ResNet_3d(Bottleneck, [3, 4, 6, 3], shortcut_type='B', no_cuda=False, num_classes=3, include_top=True)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.class_indict = {'0': 'feike', '1': 'feilian', '2': 'jinpu'}
        self.model_jishu: cv2.dnn.Net = cv2.dnn.readNetFromONNX(jishu_model_path)
        self.model_jishu.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model_jishu.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.class_indict_jishu = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}

    def fenlei(self, img_path):
        # load image
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        image_name = sort_by_numeric_value(os.listdir(img_path))

        # 使用Pillow打开图像文件
        img_1 = np.array(Image.open(os.path.join(img_path, image_name[0])))
        img_2 = np.array(Image.open(os.path.join(img_path, image_name[1])))
        img_3 = np.array(Image.open(os.path.join(img_path, image_name[2])))
        merged_array = np.zeros((128, 384, 3)).astype(np.float32)
        merged_array[:128, :128, :] = np.array(img_1).astype(np.float32)
        merged_array[:128, 128:256, :] = np.array(img_2).astype(np.float32)
        merged_array[:128, 256:, :] = np.array(img_3).astype(np.float32)
        # [N, C, H, W]
        img = self.data_transform(merged_array)
        img = torch.stack((img[:, :128, :128], img[:, :128, 128:256], img[:, :128, 256:]), dim=-1)
        img = torch.unsqueeze(img, dim=0)


        with torch.no_grad():
            # predict class
            self.model.to(self.device)
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(self.class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        # for i in range(len(predict)):
        #     print("class: {:10}   prob: {:.3}".format(self.class_indict[str(i)],
        #                                               predict[i].numpy()))

        return self.class_indict[str(predict_cla)], round(float(predict[predict_cla].numpy()), 2)

    def jishu(self, img_path):
        # load image
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        image_name = sort_by_numeric_value(os.listdir(img_path))

        original_image: np.ndarray = cv2.imread(os.path.join(img_path, image_name[2]))
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
    image_path = '/home/jjgong/PycharmProjects/colony_detecting/images/region_45'
    resnet_fenlei = fenlei_3D("./models/resnet_50.pth", "models/计数模型_三种菌.onnx")
    print(resnet_fenlei.fenlei(image_path))