import cv2
import os
import glob
import numpy as np

def nothing(x):
    pass
import cv2
import numpy as np

def set_large_regions_to_color(image, min_threshold, max_threshold, save_path):
    # 转换BGR图像到HSV空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([179, 200, 251])
    binary_image = cv2.inRange(hsv, lower_color, upper_color)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 执行形态学膨胀操作
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    binary_image = cv2.erode(binary_image, kernel, iterations=2)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)

    # 使用 connectedComponentsWithStats 找到连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    flag = False

    # 遍历每个连通区域的统计信息
    for label in range(1, num_labels):
        # 检查连通区域的面积是否超过阈值
        if stats[label, cv2.CC_STAT_AREA] > 12000:
            flag = True
        if stats[label, cv2.CC_STAT_AREA] > min_threshold and stats[label, cv2.CC_STAT_AREA] < max_threshold:
            # 生成与当前标签对应的二值图像
            region_mask = np.zeros_like(labels, dtype=np.uint8)
            region_mask[labels == label] = 255

            # 获取连通区域的坐标
            x, y, w, h = cv2.boundingRect(region_mask)

            # 从原图中切割出连通区域
            region = image[y:y+h, x:x+w]
            region = cv2.resize(region, (128, 128))

            # 保存连通区域图像
            region_filename = os.path.join(save_path, f'region_{label}.png')
            cv2.imwrite(region_filename, region)

    # 将处理后的标签图转换回彩色图像（这里实际上不需要，因为函数的目的是保存连通区域）
    return flag

def single(img_path, saved_path, image=None):
    # 读取图像
    if image is None:
        image = cv2.imread(img_path)

    # 设定面积阈值和颜色
    min_threshold = 500  # 设置阈值为1000像素
    max_threshold = 1500
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    # 标记满足条件的连通区域
    result = set_large_regions_to_color(image, min_threshold, max_threshold, saved_path)
    return result

    # # 显示结果
    # cv2.namedWindow('Result', cv2.WINDOW_FREERATIO)
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def img_path_process(img_path, saved_path):

    # 设定面积阈值和颜色
    min_threshold = 500  # 设置阈值为1000像素
    max_threshold = 1500

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    image_files = []
    # 使用 glob 模块来匹配文件夹下的所有图片文件
    image_files.extend(glob.glob(os.path.join(img_path, '*.jpg')))
    image_files.extend(glob.glob(os.path.join(img_path, '*.png')))
    for image_path in image_files:
        image = cv2.imread(image_path)
        result = set_large_regions_to_color(image, min_threshold, max_threshold, saved_path)
        # cv2.imwrite(os.path.join(saved_path, image_path.split('/')[-1]), result)


if __name__ == '__main__':
    # single_test(r'/home/jjgong/colony_count/dataset/肺克3/肺克2/230904453肺炎克雷伯菌2h/C00000213190_15_1156.jpg')
    img_path_process(r'/home/jjgong/colony_count/automatical-colony-counting/肺链-6-lhw', r'/home/jjgong/PycharmProjects/Colony/images')