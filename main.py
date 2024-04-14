import os
import shutil

from resnet_fenlei_3DConv import Fenlei3D
from yolo_inference import yolo_13h
from img_preprocess import single

temp_file_savepath = "./images"
if not os.path.exists(temp_file_savepath):
    os.mkdir(temp_file_savepath)

yolo_detect = yolo_13h("./models/分类模型_三种菌_13h.onnx", "./models/计数模型_三种菌.onnx")
# 获取单个细胞图像
is_youxiao = single('/home/jjgong/colony_count/dataset/肺克/肺克2/230301302肺炎克雷伯菌3h/C00000147166_13_1162.jpg', temp_file_savepath)

# 单个细胞图像的计数与分类
count = 0
cls_cnt = {'feike': 0, 'feilian': 0, 'jinpu': 1}
for file in os.listdir(temp_file_savepath):
    cnt, conf_cnt = yolo_detect.jishu(os.path.join(temp_file_savepath, file))
    if conf_cnt > 0.9:
        count += int(cnt)
    else:
        continue
    cls, conf_cls = yolo_detect.fenlei(os.path.join(temp_file_savepath, file))
    if conf_cls > 0.9:
        cls_cnt[cls] = cls_cnt[cls] + 1
print("有效计数个数：", count)
print("类别：", max(cls_cnt, key=cls_cnt.get))
if is_youxiao:
    print("有区域大于阈值，说明菌落样本已有效")

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