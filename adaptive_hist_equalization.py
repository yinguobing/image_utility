import numpy as np
import cv2 as cv
from file_list_generator import ListGenerator


def main():
    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(4, 4))

    # Read in images.
    target_dir = '/home/robin/Desktop/libvideo_processing/demo/image'
    file_list = ListGenerator().generate_list(target_dir, ['jpg'])

    for each_img in file_list:
        img = cv.imread(each_img)
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        v_a = clahe.apply(img_hsv[:, :, 2])
        v_e = cv.equalizeHist(img_hsv[:, :, 2])
        img_hsv_a = img_hsv.copy()
        img_hsv_e = img_hsv.copy()
        img_hsv_a[:,:,2] = v_a
        img_hsv_e[:,:,2] = v_e

        img_adpequalhist = cv.cvtColor(img_hsv_a, cv.COLOR_HSV2BGR)
        img_ehist = cv.cvtColor(img_hsv_e, cv.COLOR_HSV2BGR)

        res = np.hstack((img, img_ehist, img_adpequalhist))  # stacking images side-by-side

        # Show result.
        cv.imshow("preview", res)
        cv.waitKey()

    # Read in a video file.
    cap = cv.VideoCapture(
        '/home/robin/Desktop/libvideo_processing/demo/video.mp4')
    while True:
        _, img = cap.read()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (0, 0), img, 0.5, 0.5)

        cl1 = clahe.apply(img)
        res = np.hstack((img, cl1))  # stacking images side-by-side

        # Show result.
        cv.imshow("preview", res)
        cv.waitKey(30)


if __name__ == '__main__':
    main()
