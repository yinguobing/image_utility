"""
This script shows how to extract face area and corsponding facial
landmark points from IBUG dataset.
"""
import os

import cv2
import pts_tools as pt


DATA_DIR = "/home/robin/Documents/landmark/dataset/300VW_Dataset_2015_12_14/044"
TARGET_DIR = "/home/robin/Documents/landmark/223K/300vw"

TARGET_SIZE = 128


def read_image(point_file):
    """Read the corsponding image."""
    head, tail = os.path.split(point_file)
    image_file = tail.split('.')[-2]
    img_jpg = os.path.join(head, image_file + ".jpg")
    img_png = os.path.join(head, image_file + ".png")
    if os.path.exists(img_jpg):
        img = cv2.imread(img_jpg)
    else:
        img = cv2.imread(img_png)
    return img


def get_valid_points(box, points, target_size=TARGET_SIZE):
    """Update points locations according to new image size"""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    width = right_x - left_x
    height = bottom_y - top_y

    # Shift points first.
    for point in points:
        point[0] -= left_x
        point[1] -= top_y

    # Then scale if needed.
    if width == height:
        scale_ratio_x = scale_ratio_y = target_size / width
    else:
        scale_ratio_x = target_size / width
        scale_ratio_y = target_size / height
    for point in points:
        point[0] = point[0] * scale_ratio_x / target_size
        point[1] = point[0] * scale_ratio_y / target_size

    return points


def extract_face(image, points):
    """Extract face area from image and pts file."""
    # Get a valid face area box.
    valid_box = pt.get_valid_box(image, points)
    if valid_box is None:
        print("Opps, can not find valid box, using minimal box.")
        valid_box = pt.get_minimal_box(points)

    # Resize image if needed.
    face_image = image[valid_box[1]:valid_box[3], valid_box[0]: valid_box[2]]
    if valid_box[2] - valid_box[0] != TARGET_SIZE:
        face_image = cv2.resize(face_image, (TARGET_SIZE, TARGET_SIZE))

    # And update points location.
    valid_points = get_valid_points(valid_box, points, target_size=TARGET_SIZE)

    return face_image, valid_points


def main():
    """The main entrance"""
    # List all the pts files.
    pts_file_list = []
    for file_path, _, file_names in os.walk(DATA_DIR):
        for file_name in file_names:
            if file_name.split(".")[-1] in ["pts"]:
                pts_file_list.append(os.path.join(file_path, file_name))

    # Extract the image one by one. Use a dict to keep file count.
    counter = {'invalid': 0}

    for file_name in pts_file_list:
        # Read points and image, make sure point importing goes well.
        points = pt.read_points(file_name)
        assert len(points) == 68, "The landmarks should contain 68 points."
        image = read_image(file_name)

        # Fast check invalid pts file.
        if pt.points_are_valid(points, image) is False:
            counter['invalid'] += 1
            print("Invalid pts file, ignored:", file_name)
            continue

        # Extract face image and new points.
        face_image, new_points = extract_face(image, points)

        # Mark the result
        pt.draw_landmark_point(face_image, new_points)

        # New file to be written.
        head, tail = os.path.split(file_name)
        common_file_name = tail.split('.')[-2]
        subset_name = head.split('/')[-2]
        common_url = os.path.join(
            TARGET_DIR, "300vw-" + subset_name + "-" + common_file_name)
        image_url = common_url + ".jpg"
        print("New file saved:", image_url)

        # Save the files.
        cv2.imwrite(image_url, face_image)
        # save_point(point_normalized, point_url)

        # Preive the result
        cv2.imshow("Preview", face_image)
        cv2.waitKey(10)

    # All done, output debug info.
    print("All done! Total file: {}, invalid: {}, succeed: {}".format(
        len(pts_file_list), counter['invalid'],
        len(pts_file_list) - counter['invalid']))

if __name__ == '__main__':
    main()
