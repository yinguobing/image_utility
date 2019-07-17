from argparse import ArgumentParser

import cv2
from tqdm import tqdm

from file_list_generator import ListGenerator

parser = ArgumentParser()
parser.add_argument('dir', type=str, help='target path')
args = parser.parse_args()


def main():
    lg = ListGenerator()
    files_to_check = lg.generate_list(args.dir, ['jpg'])
    print("Total files: {}".format(len(files_to_check)))

    gray_img_list = []
    num_checked = 0
    for each_file in tqdm(files_to_check[54145:54146]):
        img = cv2.imread(each_file, cv2.IMREAD_ANYCOLOR)
        print(each_file)

        # Preview gray images.
        if len(img.shape) != 3:
            gray_img_list.append(each_file)
            cv2.imshow("gray", img)
            if cv2.waitKey(100) == 27:
                break

    print("Total gray images: {}".format(len(gray_img_list)))


if __name__ == '__main__':
    main()
