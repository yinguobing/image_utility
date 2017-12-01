import os
import sys
import cv2
from random import randrange
from subprocess import call

train_share = 857
total_share = 1000

current_path = "/home/robin/Desktop/hand_images/00_rps_image_set/scissor"

image_format_list = ["jpeg", "jpg", "gif", "png", "bmp", "tiff", "ppm", "pgm", "pbm", "pnm"]

count_images = 0
count_images_train = 0
count_images_test = 0
total_files = 0

cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Preview", 120, 120)


def generate_path(path, path_type="_generated"):
    dir_list = path.split()
    dir_list[-1] = dir_list[-1]+path_type
    separator = "/"
    return separator.join(dir_list)

train_path = generate_path(current_path, "_train")
test_path = generate_path(current_path, "_test")
os.makedirs(train_path)
os.makedirs(test_path)

for file_path, _, file_names in os.walk(current_path, followlinks=False):
    for file_name in file_names:
        if file_name.split(".")[-1] in image_format_list:
            count_images += 1
            old_file = os.path.join(file_path, file_name)

            # Generate random number range 1~total_share, which helps decide current image's belonging.
            if randrange(1, total_share) <= train_share:
                new_file = os.path.join(train_path, file_name)
                count_images_train += 1
            else:
                new_file = os.path.join(test_path, file_name)
                count_images_test += 1

            call(["cp", old_file, new_file])

            # Uncomment following code to preview images during counting.
            # print(file_name)
            # image = cv2.imread(old_file)
            # cv2.imshow("Preview", image)
            # cv2.waitKey(10)
        else:
            print("Not a image file: ", file_name)

        total_files += 1
        sys.stdout.write("\r>> Files : %2d, Images: %2d, Train images: %2d, Test images: %2d"
                         % (total_files, count_images, count_images_train, count_images_test))
        sys.stdout.flush()

print("\n Total files: %2d, Images: %2d, Train images: %2d, Test images: %2d"
      % (total_files, count_images, count_images_train, count_images_test))
