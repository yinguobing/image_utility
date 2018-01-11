"""
This script shows how to extract local area and corsponding facial
landmark points from IBUG dataset.
"""
import json
import os

import numpy as np

import cv2
import mark_detector as md
import pts_tools as pt

DATA_DIR = "/home/robin/Documents/landmark/223K"
TARGET_DIR = "/home/robin/Desktop/export"

TARGET_SIZE = 24


def read_image(img_file):
    """Read the corsponding image."""
    if os.path.exists(img_file):
        img = cv2.imread(img_file)
    return img


def extract_local_img(image, feature_index):
    """Extract face area from image."""
    # Do landmark detection first.
    marks = md.detect_marks(image, md.MARK_SESS, md.MARK_GRAPH)
    marks = marks * 128

    # Draw marks
    # md.draw_marks(image, marks)

    # Get the target point location.
    # IDX- TARGET
    # 30 - Nose tip
    # 8  - Chin
    # 36 - Left eye left corner
    # 45 - Right eye right corner
    # 48 - Left Mouth corner
    # 54 - Right mouth corner
    target_point_idx = feature_index
    target_point = marks[target_point_idx].astype(int)
    local_x = target_point[0]
    local_y = target_point[1]

    # Try to make a 24x24 square box with target point as center.
    local_box = [local_x - 12, local_y - 12,
                 local_x + 12, local_y + 12]

    # Check if local box is in image.
    if pt.box_in_image(local_box, image) is False:
        return None

    # Box ok, return it.
    return local_box


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
        feature_idx = 30
        with open(json_path) as file:
            label_marks = np.array(json.load(file), dtype=np.float32)
        label_marks = np.reshape(label_marks, (-1, 2))
        label_point = label_marks[feature_idx] * 128

        # Extract face image and new points.
        local_box = extract_local_img(image, feature_index=feature_idx)

        # Check if target label point is in box.
        if local_box is not None and pt.points_in_box([label_point], local_box):
            # Get new image
            local_img = image[local_box[1]:local_box[3],
                              local_box[0]:local_box[2]]

            # New point value
            point_normlized = [(label_point[0] - local_box[0]) / 24,
                               (label_point[1] - local_box[1]) / 24]
            # cv2.circle(local_img, (int(point_normlized[0] * 24),
            #                         int(point_normlized[1] * 24)), 1, (0, 255, 0), -1)

            # # Preview the Image.
            # preview_img = local_img.copy()
            # preview_img = cv2.resize(
            #     preview_img, (512, 512), interpolation=cv2.INTER_AREA)
            # cv2.imshow('preview', preview_img)
            # if cv2.waitKey() == 27:
            #     break

            # New file to be written.
            _, tail = os.path.split(file_name)
            common_file_name = tail.split('.')[-2]
            common_url = os.path.join(
                TARGET_DIR, 'feature30', common_file_name + '-' + str(feature_idx))

            # Save the Image.
            image_url = common_url + ".jpg"
            cv2.imwrite(image_url, local_img)

            # Save the new point location.
            csv_url = common_url + ".json"
            points_to_save = np.array(point_normlized).flatten()
            with open(csv_url, mode='w') as file:
                json.dump(list(points_to_save), file)

            print("New file saved:", image_url, csv_url, sep='\n')
        else:
            counter['invalid'] += 1

    # All done, output debug info.
    print("All done! Total file: {}, invalid: {}, succeed: {}".format(
        len(img_list), counter['invalid'],
        len(img_list) - counter['invalid']))


if __name__ == '__main__':
    main()
