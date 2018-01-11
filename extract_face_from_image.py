"""
This script shows how to extract face area and corsponding facial
landmark points from IBUG dataset.
"""
import os

import cv2
import face_detector as fd
import pts_tools as pt

DATA_DIR = "/home/robin/Documents/landmark/face_in_the_wild"
TARGET_DIR = "/home/robin/Desktop/export"

TARGET_SIZE = 128


def read_image(img_file):
    """Read the corsponding image."""
    if os.path.exists(img_file):
        img = cv2.imread(img_file)
    return img


def extract_face(file, tail, count):
    """Extract face area from image."""
    image = read_image(file)

    conf, raw_boxes = fd.get_facebox(image=image, threshold=0.9)
    # fd.draw_result(image, conf, raw_boxes)

    for box in raw_boxes:
        # Move box down.
        diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
        offset_y = int(abs(diff_height_width / 2))
        box_moved = pt.move_box(box, [0, offset_y])

        # Make box square.
        facebox = pt.get_square_box(box_moved)

        face_image = image[
            facebox[1]:facebox[3],
            facebox[0]: facebox[2]]

        # Save the Image.
        image_url = os.path.join(TARGET_DIR, str(count) + '-' + tail)
        if face_image.shape[0] * face_image.shape[1] != 0:
            preview_img = face_image.copy()
            preview_img = cv2.resize(preview_img, (512, 512))
            cv2.imshow('preview', preview_img)
            if cv2.waitKey() == 27:
                face_image = cv2.resize(face_image, (128, 128))
                cv2.imwrite(image_url, face_image)
                print("New file saved:", image_url)
                count += 1

    return face_image


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
    count = 0

    for file_name in img_list:

        # New file to be written.
        _, tail = os.path.split(file_name)

        # Extract face image and new points.
        face_image = extract_face(file_name, tail, count)

        # Preive the result
        # cv2.imshow("Preview", face_image)
        # cv2.waitKey()

    # All done, output debug info.
    print("All done! Total file: {}, invalid: {}, succeed: {}".format(
        len(img_list), counter['invalid'],
        len(img_list) - counter['invalid']))


if __name__ == '__main__':
    main()
