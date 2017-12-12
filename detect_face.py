"""
This script is slightly modified for facial landmark localization.
You can find the original code here:
https://github.com/opencv/opencv/blob/master/samples/dnn/resnet_ssd_face_python.py
"""
import cv2 as cv
from cv2 import dnn

WIDTH = 300
HEIGHT = 300
THRESHOLD = 0.5

PROTOTXT = 'face_detector/deploy.prototxt'
MODEL = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'

NET = dnn.readNetFromCaffe(PROTOTXT, MODEL)
VIDEO = '/home/robin/Documents/landmark/dataset/300VW_Dataset_2015_12_14/538/vid.avi'


def get_facebox(image=None, threshold=0.5):
    """
    Get the bounding box of faces in image.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    confidences = []
    faceboxes = []

    NET.setInput(dnn.blobFromImage(
        image, 1.0, (WIDTH, HEIGHT), (104.0, 177.0, 123.0), False, False))
    detections = NET.forward()

    for result in detections[0, 0, :, :]:
        confidence = result[2]
        if confidence > threshold:
            x_left_bottom = int(result[3] * cols)
            y_left_bottom = int(result[4] * rows)
            x_right_top = int(result[5] * cols)
            y_right_top = int(result[6] * rows)
            confidences.append(confidence)
            faceboxes.append(
                [x_left_bottom, y_left_bottom, x_right_top, y_right_top])
    return confidences, faceboxes


def draw_result(image, confidences, faceboxes):
    """Draw the detection result on image"""
    for result in zip(confidences, faceboxes):
        conf = result[0]
        facebox = result[1]

        cv.rectangle(image, (facebox[0], facebox[1]),
                     (facebox[2], facebox[3]), (0, 255, 0))
        label = "face: %.4f" % conf
        label_size, base_line = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                     (facebox[0] + label_size[0],
                      facebox[1] + base_line),
                     (0, 255, 0), cv.FILLED)
        cv.putText(image, label, (facebox[0], facebox[1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


def get_square_box(faceboxes):
    """Get the square boxes which are ready for CNN from the faceboxes"""
    square_boxes = []
    for facebox in faceboxes:
        box_width = facebox[2] - facebox[0]
        box_height = facebox[3] - facebox[1]

        diff = box_height - box_width
        delta = int(diff / 2)

        left_top_x = facebox[0] - delta
        left_top_y = facebox[1] + delta
        right_bottom_x = facebox[2] + delta
        right_bottom_y = facebox[3] + delta

        if diff % 2 == 1:
            right_bottom_x += 1

        square_boxes.append(
            [left_top_x, left_top_y, right_bottom_x, right_bottom_y])
    return square_boxes


def draw_box(image, faceboxes, box_color=(255, 255, 255)):
    """Draw square boxes on image"""
    for facebox in faceboxes:
        cv.rectangle(image, (facebox[0], facebox[1]),
                     (facebox[2], facebox[3]), box_color)


def fit_box(image, points, facebox):
    """
    Fit the box, make sure it's inside the image and contains all the points.
    """
    # First check the box if it's outside of the image.
    rows = image.shape[0]
    cols = image.shape[1]

    def _check_box(box):
        """Check if box is valid."""
        # Bounding edge of the landmark points.
        min_x = min([point[0] for point in points])
        max_x = max([point[0] for point in points])
        min_y = min([point[1] for point in points])
        max_y = max([point[1] for point in points])
        # Box is in image.
        is_in_image = box[0] >= 0 or box[1] >= 0 or box[2] <= cols or box[3] <= rows
        # Box contains all the points.
        contains_points = box[0] <= min_x or box[1] <= min_y or box[2] >= max_x or box[3] >= max_y
        # Box is square.
        w_equal_h = (box[2] - box[0]) == (box[3] - box[1])
        # Return the result.
        return is_in_image and contains_points and w_equal_h

    def _move_box(box):
        """Method 1: Try to move the box."""
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
            if _check_box([left_x, top_y, right_x, bottom_y]):
                print("Moving succeed!")
                return [left_x, top_y, right_x, bottom_y]
            else:
                print("Moving failed.")
                return None

    def _shrink_box(box):
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
        if _check_box([left_x, top_y, right_x, bottom_y]):
            print("Shrink succeed!")
            return [left_x, top_y, right_x, bottom_y]
        else:
            print("Shrink failed")
            return None

    # First try to move the box.
    box_moved = _move_box(facebox)

    # If moving faild ,try to shrink.
    if box_moved is not None:
        return box_moved
    else:
        box_shrinked = _shrink_box(facebox)

    # If shrink failed, return the original image.
    if box_shrinked is not None:
        return box_shrinked
    else:
        return [0, 0, cols, rows]


def main():
    """The main entrance"""
    cap = cv.VideoCapture(VIDEO)
    while True:
        ret, frame = cap.read()
        confidences, faceboxes = get_facebox(frame, threshold=0.5)
        draw_result(frame, confidences, faceboxes)
        cv.imshow("detections", frame)
        if cv.waitKey(1) != -1:
            break


if __name__ == '__main__':
    main()
