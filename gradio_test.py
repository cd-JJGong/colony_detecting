import gradio as gr
from yolo_inference import yolo_13h
from img_preprocess import single
import os
import shutil
import numpy as np
import cv2

temp_file_savepath = "./images"
if not os.path.exists(temp_file_savepath):
    os.mkdir(temp_file_savepath)

model = yolo_13h("./models/分类模型_三种菌_13h.onnx", "./models/计数模型_三种菌.onnx")

# 定义处理函数
def classify_image(img):
    byte_array = img.tobytes()

    # 步骤3: 获取PIL图像的尺寸和模式
    width, height = img.size
    opencv_image = np.frombuffer(byte_array, np.uint8).reshape((height, width, 3))
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    is_youxiao = single('/home/jjgong/colony_count/automatical-colony-counting/肺链-6-lhw/C00000047432_6_1145.jpg',
                        temp_file_savepath, opencv_image)
    # 单个细胞图像的计数与分类
    count = 0
    cls_cnt = {'feike': 0.0, 'feilian': 0.0, 'jinpu': 0.0}
    for file in os.listdir(temp_file_savepath):
        cls, conf_cls = model.fenlei(os.path.join(temp_file_savepath, file))
        if conf_cls > 0.9:
            cls_cnt[cls] = cls_cnt[cls] + 1
            count = count + 1
    result = [{'label': 'feike', 'score': str(cls_cnt['feike']/count)}, {'label': 'feilian', 'score': str(cls_cnt['feilian'] / count)}, {'label': 'jinpu', 'score': str(cls_cnt['jinpu'] / count)}]
    # 检查文件夹是否存在
    if os.path.exists(temp_file_savepath):
        try:
            # 使用shutil.rmtree()删除文件夹，无论是否为空
            shutil.rmtree(temp_file_savepath)
            print(f"文件夹及其内容已成功删除: {temp_file_savepath}")
        except Exception as error:
            print(f"删除文件夹时出错: {error}")
    else:
        print(f"文件夹不存在: {temp_file_savepath}")

    return {i['label']: i['score'] for i in result}

# 创建Gradio界面
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3))
iface.launch()