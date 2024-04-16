import cv2
import os
import glob
import numpy as np
from filename_sorted import sort_by_numeric_value


def nothing(x):
    pass


import cv2
import numpy as np
from filename_sorted import delete_files_in_folder


def set_large_regions_to_color(image, min_threshold, max_threshold, save_path, lower_color=np.array([0, 0, 0]), upper_color=np.array([179, 165, 251])):
    # 转换BGR图像到HSV空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binary_image = cv2.inRange(hsv, lower_color, upper_color)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 执行形态学膨胀操作
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    binary_image = cv2.erode(binary_image, kernel, iterations=2)
    binary_image = cv2.dilate(binary_image, kernel, iterations=3)

    # 使用 connectedComponentsWithStats 找到连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    flag = False
    re = False
    img_count = 0


    # 遍历每个连通区域的统计信息
    for label in range(1, num_labels):
        # 检查连通区域的面积是否超过阈值
        if stats[label, cv2.CC_STAT_AREA] > 12000:
            flag = True
        if min_threshold < stats[label, cv2.CC_STAT_AREA] < max_threshold:
            # 生成与当前标签对应的二值图像
            region_mask = np.zeros_like(labels, dtype=np.uint8)
            region_mask[labels == label] = 255

            # 获取连通区域的坐标
            x, y, w, h = cv2.boundingRect(region_mask)
            x = x
            y = y
            w = w
            h = h

            # 从原图中切割出连通区域
            region = image[y:y + h, x:x + w]
            region = cv2.resize(region, (128, 128))

            # 保存连通区域图像
            region_filename = os.path.join(save_path, f'region_{label}.png')
            img_count = img_count + 1
            cv2.imwrite(region_filename, region)
    if img_count < 10:
        re = True

    # 将处理后的标签图转换回彩色图像（这里实际上不需要，因为函数的目的是保存连通区域）
    return flag, re


def set_large_regions_to_color_3D(image_paths, min_threshold, max_threshold, save_path, lower_color=np.array([0, 0, 0]), upper_color=np.array([179, 165, 255])):
    path = os.listdir(image_paths)
    sorted_path = sort_by_numeric_value(path)
    last_img_path = sorted_path[-1]
    image = cv2.imread(os.path.join(image_paths, last_img_path))
    image_first = cv2.imread(os.path.join(image_paths, sorted_path[-3]))
    image_second = cv2.imread(os.path.join(image_paths, sorted_path[-2]))
    # 转换BGR图像到HSV空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binary_image = cv2.inRange(hsv, lower_color, upper_color)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 执行形态学膨胀操作
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    binary_image = cv2.erode(binary_image, kernel, iterations=1)
    binary_image = cv2.dilate(binary_image, kernel, iterations=3)

    # 使用 connectedComponentsWithStats 找到连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    flag = False
    img_count = 0
    re = False

    # 遍历每个连通区域的统计信息
    for label in range(1, num_labels):
        # 检查连通区域的面积是否超过阈值
        if stats[label, cv2.CC_STAT_AREA] > 12000:
            flag = True
        if min_threshold < stats[label, cv2.CC_STAT_AREA] < max_threshold:
            # 生成与当前标签对应的二值图像
            region_mask = np.zeros_like(labels, dtype=np.uint8)
            region_mask[labels == label] = 255

            # 获取连通区域的坐标
            x, y, w, h = cv2.boundingRect(region_mask)
            x = x
            y = y
            w = w
            h = h
            real_save_path = os.path.join(save_path, f'region_{label}')
            os.mkdir(real_save_path)

            # 从原图中切割出连通区域
            region = image[y:y + h, x:x + w]
            region = cv2.resize(region, (128, 128))
            # 保存连通区域图像
            region_filename = os.path.join(real_save_path, f'3.png')
            cv2.imwrite(region_filename, region)

            region = image_second[y:y + h, x:x + w]
            region = cv2.resize(region, (128, 128))
            region_filename = os.path.join(real_save_path, f'2.png')
            cv2.imwrite(region_filename, region)

            region = image_first[y:y + h, x:x + w]
            region = cv2.resize(region, (128, 128))
            region_filename = os.path.join(real_save_path, f'1.png')
            img_count = img_count + 1
            cv2.imwrite(region_filename, region)
    if img_count < 10:
        re = True
    # 将处理后的标签图转换回彩色图像（这里实际上不需要，因为函数的目的是保存连通区域）
    return flag, re


def multiple(img_path, saved_path):
    # 设定面积阈值和颜色
    min_threshold = 200  # 设置阈值为1000像素
    max_threshold = 2500
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    # 标记满足条件的连通区域
    result, re = set_large_regions_to_color_3D(img_path, min_threshold, max_threshold, saved_path)
    if re:
        delete_files_in_folder(saved_path)
        result, _ = set_large_regions_to_color_3D(img_path, min_threshold, max_threshold, saved_path, lower_color=np.array([0, 0, 0]), upper_color=np.array([179, 230, 255]))
    return result


def single(img_path, saved_path, image=None):
    # 读取图像
    if image is None:
        image = cv2.imread(img_path)

    # 设定面积阈值和颜色
    min_threshold = 200  # 设置阈值为1000像素
    max_threshold = 2500
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    # 标记满足条件的连通区域
    result, re = set_large_regions_to_color(image, min_threshold, max_threshold, saved_path)
    if re:
        delete_files_in_folder(saved_path)
        result, _ = set_large_regions_to_color(image, min_threshold, max_threshold, saved_path, lower_color=np.array([0, 0, 0]), upper_color=np.array([179, 230, 255]))
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
        result, re = set_large_regions_to_color(image, min_threshold, max_threshold, saved_path)
        # cv2.imwrite(os.path.join(saved_path, image_path.split('/')[-1]), result)


if __name__ == '__main__':
    # single_test(r'/home/jjgong/colony_count/dataset/肺克3/肺克2/230904453肺炎克雷伯菌2h/C00000213190_15_1156.jpg')
    img_path_process(r'/home/jjgong/colony_count/automatical-colony-counting/肺链-6-lhw',
                     r'/home/jjgong/PycharmProjects/Colony/images')
