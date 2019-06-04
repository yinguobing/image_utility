"""Resize all images in directory. The original images will be REPLACED"""
import os

import cv2

IMAGE_FORMATS = ["jpeg", "jpg", "gif", "png",
                 "bmp", "tiff", "ppm", "pgm", "pbm", "pnm"]

# Directory of images to be resized.
IMAGE_DIR = "/data/dataset/public/lfw/eval_112"

# To which size the image should be resized.
TARGET_SIZE = 112


def run():
    image_files_count = 0
    total_files_count = 0

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)

    for file_path, _, file_names in os.walk(IMAGE_DIR, followlinks=False):
        for file_name in file_names:
            if file_name.split(".")[-1] in IMAGE_FORMATS:
                image_files_count += 1
                # print(file_name)

                file_url = os.path.join(file_path, file_name)

                image = cv2.imread(os.path.join(file_path, file_name))
                height, width, depth = image.shape

                if height == width and height == TARGET_SIZE:
                    continue
                else:
                    image = cv2.resize(
                        image, (TARGET_SIZE, TARGET_SIZE))

                cv2.imwrite(file_url, image)
                cv2.imshow("Preview", image)
                cv2.waitKey(10)
            else:
                print("Not a image file: ", file_name)

            total_files_count += 1

    print("Total files: {}, images re-sized: {}".format(total_files_count, image_files_count))


if __name__ == "__main__":
    run()
