"""
This script shows how to generate a heat map of the facial
landmark points from IBUG dataset.
"""
import json
import math
import os

import numpy as np

import cv2
import pts_tools as pt

DATA_DIR = "/home/robin/Documents/landmark/223K/"
TARGET_DIR = "/home/robin/Desktop/export"


def read_image(img_file):
    """Read the corsponding image."""
    if os.path.exists(img_file):
        img = cv2.imread(img_file)
    return img


def put_heat(heatmap, center, sigma=3):
    """
    Put heat on image.
    This function is borrowed from:
    https://github.com/ildoonet/tf-pose-estimation
    """
    center_x, center_y = center
    height, width = heatmap.shape

    th = 4.6052
    delta = math.sqrt(th * 2)

    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))

    x1 = int(min(width, center_x + delta * sigma))
    y1 = int(min(height, center_y + delta * sigma))

    for y in range(y0, y1):
        for x in range(x0, x1):
            d = (x - center_x) ** 2 + (y - center_y) ** 2
            exp = d / 2.0 / sigma / sigma
            if exp > th:
                continue
            heatmap[y][x] = max(heatmap[y][x], math.exp(-exp))
            heatmap[y][x] = min(heatmap[y][x], 1.0)


def main():
    """The main entrance"""
    # List all the image files.
    img_list = []
    for file_path, _, file_names in os.walk(DATA_DIR):
        for file_name in file_names:
            if file_name.split(".")[-1] in ["jpg"]:
                img_list.append(os.path.join(file_path, file_name))

    # Extract the image one by one. Use a dict to keep file count.
    counter = {'invalid': 0}

    for file_name in img_list:
        print(file_name)
        # Read in image file.
        image = read_image(file_name)

        # Read in label point.
        json_path = file_name.split('.')[-2] + '.json'
        with open(json_path) as file:
            label_marks = np.array(json.load(file), dtype=np.float32)
        label_marks = np.reshape(label_marks, (-1, 2)) * 128

        # Draw heat on map.
        heat_map = np.zeros((128, 128), dtype=np.float32)
        for point in label_marks:
            put_heat(heat_map, point)

        # Preview heatmap.
        heatmap = cv2.cvtColor(heat_map, cv2.COLOR_GRAY2BGR)
        heatmap *= 255
        heatmap = np.uint8(heatmap)
        heatmap = cv2.resize(heatmap, (64, 64), interpolation=cv2.INTER_AREA)
        heatmap = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_AREA)

        # Preview the Image.
        preview_img = image.copy()
        preview_img = cv2.resize(
            preview_img, (512, 512), interpolation=cv2.INTER_AREA)

        img_to_show = np.concatenate((heatmap, preview_img), axis=1)
        cv2.imshow('preview', img_to_show)
        if cv2.waitKey(15) == 27:
            break

        # New file to be written.
        # _, tail = os.path.split(file_name)
        # common_file_name = tail.split('.')[-2]
        # common_url = os.path.join(
        #     TARGET_DIR, 'feature30', common_file_name + '-' + str(feature_idx))

        # Save the Image.
        # image_url = common_url + ".jpg"
        # cv2.imwrite(image_url, local_img)

        # print("New file saved:", image_url, sep='\n')

    # All done, output debug info.
    print("All done! Total file: {}, invalid: {}, succeed: {}".format(
        len(img_list), counter['invalid'],
        len(img_list) - counter['invalid']))


if __name__ == '__main__':
    main()
