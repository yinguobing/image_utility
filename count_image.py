import os
import cv2

image_format_list = ["jpeg", "jpg", "gif", "png", "bmp", "tiff", "ppm", "pgm", "pbm", "pnm"]

current_path = "/home/robin/Desktop/hand_images/00_rps_image_set/test"

count = 0
total_files = 0

cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Preview", 120, 120)

for file_path, _, file_names in os.walk(current_path, followlinks=False):
    for file_name in file_names:
        if file_name.split(".")[-1] in image_format_list:
            count += 1

            # Uncomment following code to preview images during counting.
            # print(file_name)
            # image = cv2.imread(os.path.join(file_path, file_name))
            # cv2.imshow("Preview", image)
            # cv2.waitKey(10)
        else:
            print("Not a image file: ", file_name)

        total_files += 1

print("Total files: %2d, Images: %2d" % (total_files, count))
