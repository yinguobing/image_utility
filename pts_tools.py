"""
This script shows how to read iBUG pts file and draw all the landmark points on image.
"""

import os

import numpy as np

import cv2
import detect_face as fd

DATA_DIR = "/home/robin/Documents/landmark/dataset"


def read_points(file_name=None):
    """
    Read points from .pts file.
    """
    points = []
    with open(file_name) as file:
        line_count = 0
        for line in file:
            if "version" in line or "points" in line or "{" in line or "}" in line:
                continue
            else:
                x, y = line.strip().split(sep=" ")
                points.append([float(x), float(y)])
                line_count += 1
    return points


def draw_landmark_point(image, points):
    """
    Draw landmark point on image.
    """
    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)


def preview(point_file):
    """
    Preview points on image.
    """
    # Read the points from file.
    raw_points = read_points(point_file)

    # Safe guard, make sure point importing goes well.
    assert len(raw_points) == 68, "The landmarks should contain 68 points."

    # Read the image.
    head, tail = os.path.split(point_file)
    image_file = tail.split('.')[-2]
    img_jpg = os.path.join(head, image_file+".jpg")
    img_png = os.path.join(head, image_file+".png")
    if os.path.exists(img_jpg):
        img = cv2.imread(img_jpg)
    else:
        img = cv2.imread(img_png)

    # Get the face bounding boxes.
    conf, faceboxes = fd.get_facebox(img, threshold=0.9)
    fd.draw_result(img, conf, faceboxes)

    # Get the square boxs contains face.
    square_boxes = fd.get_square_box(faceboxes)

    # Remove false positive boxes.
    min_x = min([point[0] for point in raw_points])
    max_x = max([point[0] for point in raw_points])
    min_y = min([point[1] for point in raw_points])
    max_y = max([point[1] for point in raw_points])
    for idx, box in enumerate(square_boxes):
        if box[0] > min_x or box[1] > min_y or box[2] < max_x or box[3] < max_y:
            del square_boxes[idx]
    fd.draw_box(img, square_boxes)

    # Check if fitting required.
    rows = img.shape[0]
    cols = img.shape[1]
    for box in square_boxes:
        if box[0] < 0 or box[1] < 0 or box[2] > cols or box[3] > rows:
            fited_box = fd.fit_box(img, raw_points, square_boxes[0])
            if fited_box is not None:
                fd.draw_box(img, [fited_box], box_color=(255, 0, 0))

    # Draw the landmark points.
    draw_landmark_point(img, raw_points)

    # Show in window.
    width, height = img.shape[:2]
    max_height = 640
    if height > max_height:
        img = cv2.resize(img, (max_height, int(width * max_height / height)))
    cv2.imshow("preview", img)
    cv2.waitKey(30)


def main():
    """
    The main entrance
    """
    # List all the files
    pts_file_list = []
    for file_path, _, file_names in os.walk(DATA_DIR):
        for file_name in file_names:
            if file_name.split(".")[-1] in ["pts"]:
                pts_file_list.append(os.path.join(file_path, file_name))

    # Show the image one by one.
    for file_name in pts_file_list:
        preview(file_name)


if __name__ == "__main__":
    main()
