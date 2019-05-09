import json
import logging
from tqdm import tqdm

import cv2

logging.basicConfig(level=logging.DEBUG)


def main():
    # Read in image list to be converted.
    with open('gray.json', 'r') as fp:
        img_list = json.load(fp)
    logging.debug("Total files to be converted: {}".format(len(img_list)))

    # Convert them into 3 channel images.
    for each_file in tqdm(img_list):
        img = cv2.imread(each_file, cv2.IMREAD_ANYCOLOR)
        if len(img.shape) == 3:
            print("Not a gray image: {}".format(each_file))
            continue

        cv2.imshow('preview', img)
        if cv2.waitKey(30) == 27:
            break

        # Do convertion
        img_converted = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Write to file.
        cv2.imwrite(each_file, img_converted)

        # Check if convertion failed.
        img = cv2.imread(each_file, cv2.IMREAD_ANYCOLOR)
        assert len(img.shape) == 3, "Convertion failed: {}".format(each_file)


if __name__ == '__main__':
    main()
