from os import path
from pprint import pprint

import cv2
import hdf5storage
import numpy as np
from tqdm import tqdm

MAFA_ROOT = '/data/dataset/public/MAFA'
MAT_FILE = path.join(MAFA_ROOT, 'train/LabelTrainAll.mat')
EXPORT_DIR = '/data/dataset/public/MAFA/tfrecord'


def load_labels(label_file):
    out = hdf5storage.loadmat(MAT_FILE)
    record = np.array(out['label_train'])[0]
    train_samples = []
    for item in record:
        train_samples.append(
            {
                'image_file': item[1][0],
                'lables': item[2][0].astype(int)
            }
        )
    return train_samples


def parse_train_labels(raw_Labels):
    """
    raw labels form: [x,y,w,h, x1,y1,x2,y2, x3,y3,w3,h3, occ_type, occ_degree,
    gender, race, orientation, x4,y4,w4,h4]
        (a) (x,y,w,h) is the bounding box of a face,
        (b) (x1,y1,x2,y2) is the position of two eyes.
        (c) (x3,y3,w3,h3) is the bounding box of the occluder. Note that (x3,y3)
            is related to the face bounding box position (x,y)
        (d) occ_type stands for the occluder type and has: 1 for simple, 2 for
            complex and 3 for human body.
        (e) occ_degree stands for the number of occluded face parts
        (f) gender and race stand for the gender and race of one face
        (g) orientation stands for the face orientation/pose, and has: 1-left,
            2-left frontal, 3-frontal, 4-right frontal, 5-right
        (h) (x4,y4,w4,h4) is the bounding box of the glasses and is set to
            (-1,-1,-1,-1) when no glasses. Note that (x4,y4) is related to the
            face bounding box position (x,y)
    """
    return {
        'face': [raw_Labels[0], raw_Labels[1], raw_Labels[2], raw_Labels[3]],
        'eyes': [raw_Labels[4], raw_Labels[5], raw_Labels[6], raw_Labels[7]],
        'occlude': {
            'location': [raw_Labels[8], raw_Labels[9], raw_Labels[10], raw_Labels[11]],
            'type': raw_Labels[12],
            'degree': raw_Labels[13]},
        'gender': raw_Labels[14],
        'race': raw_Labels[15],
        'orientation': raw_Labels[16],
        'glass': [raw_Labels[17], raw_Labels[18], raw_Labels[19], raw_Labels[20]]
    }


def draw_face(image, labels, color=(0, 255, 0)):
    x, y, w, h = labels['face']
    cv2.rectangle(image, (x, y, w, h), color, 2)


def draw_mask(image, labels, color=(0, 255, 0)):
    x, y, w, h = labels['face']
    _x, _y, _w, _h = labels['occlude']['location']
    cv2.rectangle(image, (x + _x, y + _y),
                  (x + _w, y + _h), colors[1], 2)


def export_face(image, labels, export_file, target_size=112):
    # Crop the face
    x, y, w, h = labels['face']
    image_face = image[y:(y+h), x: (x+w)]

    # Resize and save image.
    image_face = cv2.resize(image_face, (target_size, target_size))
    cv2.imwrite(export_file, image_face)


if __name__ == "__main__":
    # Load annotations from the mat file.
    train_samples = load_labels(MAT_FILE)

    # loop through all the annotations and do processing. Here we are going to
    # extract all the occluded faces and save them in a new image file.
    for sample in tqdm(train_samples):
        labels = parse_labels(sample['lables'])
        image = cv2.imread(
            path.join(MAFA_ROOT, 'train/images', sample['image_file']))

        face_image_file = path.join(
            EXPORT_DIR, 'images', sample['image_file'])
        export_face(image, labels, face_image_file, 112)
