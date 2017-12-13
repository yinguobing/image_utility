"""
This script shows how to read iBUG pts file and draw all the landmark points on image.
"""

import os

import numpy as np

import cv2
import face_detector as fd

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


def get_minimal_box(points):
    """Get the minimal bounding box of a group of points"""
    min_x = min([point[0] for point in points])
    max_x = max([point[0] for point in points])
    min_y = min([point[1] for point in points])
    max_y = max([point[1] for point in points])
    return [min_x, min_y, max_x, max_y]


def points_in_box(points, box):
    """Check if box contains all the points"""
    minimal_box = get_minimal_box(points)
    return box[0] > minimal_box[0] or box[1] > minimal_box[1] or box[2] < minimal_box[2] or box[3] < minimal_box[3]


def box_in_image(box, image):
    """Check if the box is in image"""
    rows = image.shape[0]
    cols = image.shape[1]
    return box[0] >= 0 or box[1] >= 0 or box[2] <= cols or box[3] <= rows


def box_is_valid(image, points, box):
    """Check if box is valid."""
    # Box contains all the points.
    points_is_in_box = points_in_box(points, box)

    # Box is in image.
    box_is_in_image = box_in_image(box, image)

    # Box is square.
    w_equal_h = (box[2] - box[0]) == (box[3] - box[1])

    # Return the result.
    return box_is_in_image and points_is_in_box and w_equal_h


def fit_by_moving(box, image, points):
    """Method 1: Try to move the box."""
    rows = image.shape[0]
    cols = image.shape[1]
    # Face box points.
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]
    if right_x - left_x <= cols and bottom_y - top_y <= rows:
        if left_x < 0:                  # left edge crossed, move right.
            right_x += abs(left_x)
            left_x = 0
        if right_x > cols:              # right edge crossed, move left.
            left_x -= (right_x - cols)
            right_x = cols
        if top_y < 0:                   # top edge crossed, move down.
            bottom_y += abs(top_y)
            top_y = 0
        if bottom_y > rows:             # bottom edge crossed, move up.
            top_y -= (bottom_y - rows)
            bottom_y = rows
        # Check if method 1 suceed.
        if box_is_valid(image, points, [left_x, top_y, right_x, bottom_y]):
            return [left_x, top_y, right_x, bottom_y]
        else:
            return None


def fit_by_shrinking(box, image, points):
    """Method 2: Try to shrink the box."""
    # Face box points.
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]
    # The first step would be get the interlaced area.
    if left_x < 0:                  # left edge crossed, set zero.
        left_x = 0
    if right_x > cols:              # right edge crossed, set max.
        right_x = cols
    if top_y < 0:                   # top edge crossed, set zero.
        top_y = 0
    if bottom_y > rows:             # bottom edge crossed, set max.
        bottom_y = rows

    # Then found out which is larger: the width or height. This will
    # be used to decide in which dimention the size would be shrinked.
    width = right_x - left_x
    height = bottom_y - top_y
    delta = abs(width - height)
    if width > height:                  # x should be altered.
        if left_x != 0 and right_x != cols:     # shrink from center.
            left_x += int(delta / 2)
            right_x -= int(delta / 2)
            if delta % 2 == 1:
                right_x += 1
        elif left_x == 0:                       # shrink from right.
            right_x -= delta
        else:                                   # shrink from left.
            left_x += delta
    else:                               # y should be altered.
        if top_y != 0 and bottom_y != rows:     # shrink from center.
            top_y += int(delta / 2)
            bottom_y -= int(delta / 2)
            if delta % 2 == 1:
                top_y += 1
        elif top_y == 0:                        # shrink from bottom.
            bottom_y -= delta
        else:                                   # shrink from top.
            top_y += delta

    # Check if method 1 suceed.
    if box_is_valid(image, points, [left_x, top_y, right_x, bottom_y]):
        return [left_x, top_y, right_x, bottom_y]
    else:
        return None


def fit_box(box, image, points):
    """Try to fit the box, make sure it satisfy following conditions:
    - A square.
    - Inside the image.
    - Contains all the points.
    """
    # First try to move the box.
    box_moved = fit_by_moving(box, image, points)

    # If moving faild ,try to shrink.
    if box_moved is not None:
        print("Moving succeed!")
        return box_moved
    else:
        print("Moving failed.")
        box_shrinked = fit_by_shrinking(box, image, points)

    # If shrink failed, return the original image.
    if box_shrinked is not None:
        print("Shrinking succeed!")
        return box_shrinked
    else:
        print("Shrink failed, using minimal bounding box.")
        return get_minimal_box(points)


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
    img_jpg = os.path.join(head, image_file + ".jpg")
    img_png = os.path.join(head, image_file + ".png")
    if os.path.exists(img_jpg):
        img = cv2.imread(img_jpg)
    else:
        img = cv2.imread(img_png)

    # Get the face bounding boxes.
    conf, faceboxes = fd.get_facebox(img, threshold=0.5)
    # fd.draw_result(img, conf, faceboxes)

    # Get the square boxs contains face.
    square_boxes = fd.get_square_boxes(faceboxes)

    # Remove false positive boxes.
    valid_box = None
    for box in square_boxes:
        if points_in_box(raw_points, box):
            valid_box = box
            # fd.draw_box(img, [valid_box])

    # Draw the landmark points.
    draw_landmark_point(img, raw_points)

    # Check if fitting required.
    rows = img.shape[0]
    cols = img.shape[1]
    if valid_box is not None:
        if valid_box[0] < 0 or valid_box[1] < 0 or valid_box[2] > cols or valid_box[3] > rows:
            fited_box = fd.fit_box(img, raw_points, valid_box)
            if fited_box is not None:
                # fd.draw_box(img, [fited_box], box_color=(255, 0, 0))
                face_area = img[fited_box[1]:fited_box[3],
                                fited_box[0]:fited_box[2]]
                area = cv2.resize(face_area, (512, 512))
                cv2.imshow("face", area)
                cv2.waitKey(30)
        else:
            valid_area = img[valid_box[1]:valid_box[3],
                             valid_box[0]:valid_box[2]]
            area = cv2.resize(valid_area, (512, 512))
            cv2.imshow("face", area)
            cv2.waitKey(30)
    else:
        print("Non-valid image:", head + tail)

    # Show in window.
    # if len(square_boxes) != 1:
    #     width, height = img.shape[:2]
    #     max_height = 640
    #     if height > max_height:
    #         img = cv2.resize(
    #             img, (max_height, int(width * max_height / height)))
    #     cv2.imshow("preview", img)
    #     cv2.waitKey()


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
