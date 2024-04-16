import gradio as gr
from yolo_inference import yolo_13h
from img_preprocess import single
from img_preprocess import multiple
import os
import shutil
import numpy as np
import cv2
import zipfile
from renet50_inference import fenlei_3D
from filename_sorted import sort_by_numeric_value

temp_file_savepath = "./images"
zip_temp_file_savepath = "./temp_imgs"
if not os.path.exists(temp_file_savepath):
    os.mkdir(temp_file_savepath)

model = yolo_13h("./models/分类模型_三种菌_13h.onnx", "./models/计数模型_三种菌.onnx")
res_model = fenlei_3D("./models/resnet_50.pth", "./models/计数模型_三种菌.onnx")


# 定义处理函数
def classify_image(file_path):
    # 检查文件夹是否存在
    if os.path.exists(zip_temp_file_savepath):
        try:
            # 使用shutil.rmtree()删除文件夹，无论是否为空
            shutil.rmtree(zip_temp_file_savepath)
            print(f"文件夹及其内容已成功删除: {zip_temp_file_savepath}")
        except Exception as error:
            print(f"删除文件夹时出错: {error}")
    else:
        print(f"文件夹不存在: {zip_temp_file_savepath}")

    cls_cnt = {'feike': 0.0, 'feilian': 0.0, 'jinpu': 0.0}
    # 单个细胞图像的计数与分类
    count = 0
    sum_count = 0
    is_youxiao = False
    if file_path.split('.')[-1] == 'jpg' or file_path.split('.')[-1] == 'png':
        is_youxiao = single(file_path,
                            temp_file_savepath)

        for file in os.listdir(temp_file_savepath):
            cls, conf_cls = model.fenlei(os.path.join(temp_file_savepath, file))
            cnt, conf_cnt = model.jishu(os.path.join(temp_file_savepath, file))
            if conf_cnt > 0.8:
                sum_count = sum_count + int(cnt)
            if conf_cls > 0.7:
                cls_cnt[cls] = cls_cnt[cls] + 1
                count = count + 1

        imgs_List = [(file_path, '13小时图像')]


    elif file_path.split('.')[-1] == 'zip':
        # 指定解压目录
        extractDir = zip_temp_file_savepath

        # 确保解压目录存在
        if not os.path.exists(extractDir):
            os.makedirs(extractDir)

        # 创建一个ZipFile对象
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # 解压所有文件到指定目录
            zip_ref.extractall(extractDir)

        is_youxiao = multiple(extractDir, temp_file_savepath)
        for file in os.listdir(temp_file_savepath):
            cls, conf_cls = res_model.fenlei(os.path.join(temp_file_savepath, file))
            cnt, conf_cnt = model.jishu(os.path.join(temp_file_savepath, file, sort_by_numeric_value(os.listdir(os.path.join(temp_file_savepath, file)))[-1]))
            if conf_cnt > 0.8:
                sum_count = sum_count + int(cnt)
            if conf_cls > 0.9:
                cls_cnt[cls] = cls_cnt[cls] + 1
                count = count + 1

        imgs_List = [(os.path.join(zip_temp_file_savepath, name), leibie) for name, leibie in zip(sort_by_numeric_value(os.listdir(extractDir)), ['5小时图像', '9小时图像', '11小时图像'])]
    if count != 0 and (sum_count > 10 or is_youxiao[0]):
        result = [{'label': '肺炎克雷伯菌', 'score': str(cls_cnt['feike'] / count)},
                  {'label': '肺炎链球菌', 'score': str(cls_cnt['feilian'] / count)},
                  {'label': '金黄色葡萄球菌', 'score': str(cls_cnt['jinpu'] / count)},
                  {'label': '无效检测', 'score': '0'}]
    else:
        result = [{'label': '肺炎克雷伯菌', 'score': '0'},
                  {'label': '肺炎链球菌', 'score': '0'},
                  {'label': '金黄色葡萄球菌', 'score': '0'},
                  {'label': '无效检测', 'score': '1'}]

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

    # 检查文件夹是否存在
    # if os.path.exists(zip_temp_file_savepath):
    #     try:
    #         # 使用shutil.rmtree()删除文件夹，无论是否为空
    #         shutil.rmtree(zip_temp_file_savepath)
    #         print(f"文件夹及其内容已成功删除: {zip_temp_file_savepath}")s
    #     except Exception as error:
    #         print(f"删除文件夹时出错: {error}")
    # else:
    #     print(f"文件夹不存在: {zip_temp_file_savepath}")

    return {i['label']: i['score'] for i in result}, imgs_List


# 创建Gradio界面
iface = gr.Interface(
    fn=classify_image,
    title='基于3D卷积神经网络结合BD kiestra系统实现感染性疾病病原菌的快速鉴定',
    inputs=gr.File(file_count="single", label="欢迎使用，请上传单张第13h的生长图像或者包含三张时序图像（5h、9h、11h）的压缩包"),
    outputs=[gr.Label(num_top_classes=4, label="检测结果"), gr.Gallery(
            label="上传的图像")],
    css="""body {
    background-color: black;
    }"""
)
iface.launch(
)
