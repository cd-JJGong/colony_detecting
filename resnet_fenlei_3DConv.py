import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from three_DConv import ResNet_3d
from three_DConv import Bottleneck

from torchvision.models import resnet50
import numpy as np


class Fenlei3D:
    def __init__(self, model_path):
        self.data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ])
        self.model = ResNet_3d(Bottleneck, [3, 4, 6, 3], shortcut_type='B', no_cuda=False, num_classes=3, include_top=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.class_indict = {'0': 'feike', '1': 'feilian', '2': 'jinpu'}

    def fenlei(self, img_path):
        # load image
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        image_name = os.listdir(img_path)
        # 使用Pillow打开图像文件
        img_1 = np.array(Image.open(os.path.join(img_path, image_name[0]))).astype(np.float32)
        img_2 = np.array(Image.open(os.path.join(img_path, image_name[1]))).astype(np.float32)
        img_3 = np.array(Image.open(os.path.join(img_path, image_name[2]))).astype(np.float32)
        merged_array = np.zeros((384, 384, 3)).astype(np.float32)
        merged_array[:128, :128, :] = np.array(img_1).astype(np.float32)
        merged_array[:128, 128:256, :] = np.array(img_2).astype(np.float32)
        merged_array[:128, 256:, :] = np.array(img_3).astype(np.float32)
        # [N, C, H, W]
        img = self.data_transform(merged_array)
        img = torch.stack((img[ :, :128, :128], img[ :, :128, 128:256], img[ :, :128, 256:]), dim=-1)
        img = torch.unsqueeze(img, dim=0)
        # print(img.shape)

        # prediction

        with torch.no_grad():
            # predict class
            self.model.to(self.device)
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        # print_res = "class: {}   prob: {:.3}".format(self.class_indict[str(predict_cla)],
        #                                              predict[predict_cla].numpy())
        # plt.title(print_res)
        # for i in range(len(predict)):
        #     print("class: {:10}   prob: {:.3}".format(self.class_indict[str(i)],
        #                                               predict[i].numpy()))

        return self.class_indict[str(predict_cla)]


if __name__ == '__main__':
    img_path = r"/home/jjgong/colony_count/最终数据集/3D分类数据集/3Dfenlei/train/jinpu/0_33"
    model_path = "models/resnet_50.pth"
    fenlei = Fenlei3D(model_path)
    print(fenlei.fenlei(img_path))