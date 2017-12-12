"""
This script shows how to read iBUG pts file and draw all the landmark points on image.
"""

import os

import numpy as np

import cv2
import detect_face as fd

IMG_DIR = "/home/robin/Documents/landmark/dataset/lfpw/testset"
IMG_FORMAT = ".png"


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
    raw_points = read_points(os.path.join(IMG_DIR, point_file))

    # Safe guard, make sure point importing goes well.
    assert len(raw_points) == 68, "The landmarks should contain 68 points."

    # Read the image.
    img = cv2.imread(os.path.join(
        IMG_DIR, point_file.split(".")[-2] + IMG_FORMAT))

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
    cv2.waitKey()


def main():
    """
    The main entrance
    """
    # List all the files
    for _, _, file_names in os.walk(IMG_DIR):
        pts_file_list = [
            file for file in file_names if file.split(".")[-1] in ["pts"]]

    # Show the image one by one.
    for file in pts_file_list:
        preview(file)


if __name__ == "__main__":
    main()
