"""
This script shows how to read iBUG pts file and draw all the landmark points on image.
"""

import os

import cv2
import face_detector as fd

DATA_DIR = "/home/robin/Documents/landmark/dataset/"


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
                loc_x, loc_y = line.strip().split(sep=" ")
                points.append([float(loc_x), float(loc_y)])
                line_count += 1
    return points


def draw_landmark_point(image, points):
    """
    Draw landmark point on image.
    """
    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)


def points_are_valid(points, image):
    """Check if all points are in image"""
    min_box = get_minimal_box(points)
    if box_in_image(min_box, image):
        return True
    return False


def get_square_box(box):
    """Get the square boxes which are ready for CNN from the boxes"""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   # Already a square.
        return box
    elif diff > 0:                  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [left_x, top_y, right_x, bottom_y]


def get_minimal_box(points):
    """
    Get the minimal bounding box of a group of points.
    The coordinates are also converted to int numbers.
    """
    min_x = int(min([point[0] for point in points]))
    max_x = int(max([point[0] for point in points]))
    min_y = int(min([point[1] for point in points]))
    max_y = int(max([point[1] for point in points]))
    return [min_x, min_y, max_x, max_y]


def move_box(box, offset):
    """Move the box to direction specified by offset"""
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]
    return [left_x, top_y, right_x, bottom_y]


def expand_box(square_box, scale_ratio=1.2):
    """Scale up the box"""
    assert (scale_ratio >= 1), "Scale ratio should be greater than 1."
    delta = int((square_box[2] - square_box[0]) * (scale_ratio - 1) / 2)
    left_x = square_box[0] - delta
    left_y = square_box[1] - delta
    right_x = square_box[2] + delta
    right_y = square_box[3] + delta
    return [left_x, left_y, right_x, right_y]


def points_in_box(points, box):
    """Check if box contains all the points"""
    minimal_box = get_minimal_box(points)
    return box[0] <= minimal_box[0] and \
        box[1] <= minimal_box[1] and \
        box[2] >= minimal_box[2] and \
        box[3] >= minimal_box[3]


def box_in_image(box, image):
    """Check if the box is in image"""
    rows = image.shape[0]
    cols = image.shape[1]
    return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows


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


def fit_by_shifting(box, rows, cols):
    """Method 1: Try to move the box."""
    # Face box points.
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    # Check if moving is possible.
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

    return [left_x, top_y, right_x, bottom_y]


def fit_by_shrinking(box, rows, cols):
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
    # Find out which dimention should be altered.
    if width > height:                  # x should be altered.
        if left_x != 0 and right_x != cols:     # shrink from center.
            left_x += int(delta / 2)
            right_x -= int(delta / 2) + delta % 2
        elif left_x == 0:                       # shrink from right.
            right_x -= delta
        else:                                   # shrink from left.
            left_x += delta
    else:                               # y should be altered.
        if top_y != 0 and bottom_y != rows:     # shrink from center.
            top_y += int(delta / 2) + delta % 2
            bottom_y -= int(delta / 2)
        elif top_y == 0:                        # shrink from bottom.
            bottom_y -= delta
        else:                                   # shrink from top.
            top_y += delta

    return [left_x, top_y, right_x, bottom_y]


def fit_box(box, image, points):
    """
    Try to fit the box, make sure it satisfy following conditions:
    - A square.
    - Inside the image.
    - Contains all the points.
    If all above failed, return None.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    # First try to move the box.
    box_moved = fit_by_shifting(box, rows, cols)

    # If moving faild ,try to shrink.
    if box_is_valid(image, points, box_moved):
        return box_moved
    else:
        box_shrinked = fit_by_shrinking(box, rows, cols)

    # If shrink failed, return None
    if box_is_valid(image, points, box_shrinked):
        return box_shrinked

    # Finally, Worst situation.
    print("Fitting failed!")
    return None


def get_valid_box(image, points):
    """
    Try to get a valid face box which meets the requirments.
    The function follows these steps:
        1. Try method 1, if failed:
        2. Try method 0, if failed:
        3. Return None
    """
    # Try method 1 first.
    def _get_postive_box(raw_boxes, points):
        for box in raw_boxes:
            # Move box down.
            diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            offset_y = int(abs(diff_height_width / 2))
            box_moved = move_box(box, [0, offset_y])

            # Make box square.
            square_box = get_square_box(box_moved)

            # Remove false positive boxes.
            if points_in_box(points, square_box):
                return square_box
        return None

    # Try to get a positive box from face detection results.
    _, raw_boxes = fd.get_facebox(image, threshold=0.5)
    positive_box = _get_postive_box(raw_boxes, points)
    if positive_box is not None:
        if box_in_image(positive_box, image) is True:
            return positive_box
        return fit_box(positive_box, image, points)

    # Method 1 failed, Method 0
    min_box = get_minimal_box(points)
    sqr_box = get_square_box(min_box)
    epd_box = expand_box(sqr_box)
    if box_in_image(epd_box, image) is True:
        return epd_box
    return fit_box(epd_box, image, points)


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

    # Fast check: all points are in image.
    if points_are_valid(raw_points, img) is False:
        return None

    # Get the valid facebox.
    facebox = get_valid_box(img, raw_points)
    if facebox is None:
        print("Using minimal box.")
        facebox = get_minimal_box(raw_points)
    # fd.draw_box(img, [facebox], box_color=(255, 0, 0))

    # Extract valid image area.
    face_area = img[facebox[1]:facebox[3],
                    facebox[0]: facebox[2]]

    # Check if resize is needed.
    width = facebox[2] - facebox[0]
    height = facebox[3] - facebox[1]
    if width != height:
        print('opps!', width, height)
    if (width != 128) or (height != 128):
        face_area = cv2.resize(face_area, (256, 256))

    # Show the result.
    cv2.imshow("face", face_area)
    if cv2.waitKey(10) == 27:
        cv2.waitKey()

    # Show whole image in window.
    # width, height = img.shape[:2]
    # max_height = 640
    # if height > max_height:
    #     img = cv2.resize(
    #         img, (max_height, int(width * max_height / height)))
    # cv2.imshow("preview", img)
    # cv2.waitKey()


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
