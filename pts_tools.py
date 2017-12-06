import os
import numpy as np
import cv2

IMG_DIR = "/home/robin/Documents/landmark/dataset/ibug"
IMG_FORMAT = ".jpg"


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

    # Draw the landmark points.
    draw_landmark_point(img, raw_points)

    # Show in window.
    width, height = img.shape[:2]
    max_height = 640
    if height > max_height:
        img = cv2.resize(img, (max_height, int(width*max_height/height)))
    cv2.imshow("preview", img)
    cv2.waitKey(100)


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
