import glob
import os
import xml.etree.ElementTree as ET

import pandas as pd

import cv2


def get_bounding_box_from(xml_file, record_count):
    """
    Find the hand bounding box from xml file and apped to xml value with
     meta info.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    records = []
    for member in root.findall('object'):
        if member[0].text == "person":
            for human_part in member.findall('part'):
                if human_part[0].text == "hand":
                    record = (root.find('filename').text,
                              int(root.find('size').find('width').text),
                              int(root.find('size').find('height').text),
                              human_part[0].text,
                              int(human_part[1][0].text),
                              int(human_part[1][1].text),
                              int(human_part[1][2].text),
                              int(human_part[1][3].text))
                    records.append(record)
                    record_count += 1
    return (records, record_count)


def run():
    image_format_list = ["jpeg", "jpg", "gif", "png",
                         "bmp", "tiff", "ppm", "pgm", "pbm", "pnm"]

    current_path = "/home/robin/Desktop/SSD/palm-dataset/data/VOCdevkit/VOC2012/"

    image_count = 0
    hand_count = 0
    hand_image_count = 0
    total_files = 0

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Preview", 120, 120)

    # Walk through directories for image files.
    record_list = []
    for file_path, _, file_names in os.walk(current_path + "JPEGImages", followlinks=False):
        for file_name in file_names:
            if file_name.split(".")[-1] in image_format_list:
                print(file_name)
                image_count += 1

                # Get annotation
                xml_file = os.path.join(
                    current_path, "Annotations", file_name.split(".")[-2] + ".xml")
                current_records, hand_count = get_bounding_box_from(
                    xml_file, hand_count)

                # Try to update record list
                if current_records != []:
                    record_list += current_records
                    hand_image_count += 1

                    # Uncomment following code to preview images during counting.
                    # image = cv2.imread(os.path.join(file_path, file_name))
                    
                    # # Draw the bounding box on image
                    # def draw_box(image, records):
                    #     print(records)
                    #     for record in records:
                    #         pt1 = record[4]
                    #         pt2 = record[5]
                    #         pt3 = record[6]
                    #         pt4 = record[7]
                    #         cv2.rectangle(image, (pt1, pt2), (pt3, pt4), (0, 255, 0), 3)

                    # draw_box(image, current_records)

                    # cv2.imshow("Preview", image)
                    # c = cv2.waitKey(50)
                    # if c == ord('q'):
                    #     print("Terminated by user.")
                    #     exit()
            else:
                print("Not a image file: ", file_name)

            total_files += 1

    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']

    # Save list into csv file.
    xml_df = pd.DataFrame(record_list, columns=column_name)
    xml_df.to_csv('hand_labels.csv', index=None)

    print("Total files: %2d, Images: %2d, , Hand images: %2d, Hands: %2d" %
          (total_files, image_count, hand_image_count, hand_count))
    print("csv file saved as \"hand_labels.csv\"")


def main():
    run()


if __name__ == '__main__':
    main()