from os import path
from pprint import pprint

import cv2
import hdf5storage
import numpy as np
from tqdm import tqdm

MAFA_ROOT = '/data/dataset/public/MAFA'
MAT_FILE = path.join(MAFA_ROOT, 'test/LabelTestAll.mat')
EXPORT_DIR = '/data/dataset/public/MAFA/tfrecord/test_images'


def load_labels(label_file, data_type='train'):
    out = hdf5storage.loadmat(MAT_FILE)
    samples = []
    if data_type == 'train':
        record = np.array(out['label_train'])[0]
        for item in record:
            samples.append(
                {
                    'image_file': item[1][0],
                    'lables': item[2][0].astype(int)
                }
            )
    else:
        record = np.array(out['LabelTest'])[0]
        for item in record:
            samples.append(
                {
                    'image_file': item[0][0],
                    'lables': item[1][0].astype(int)
                }
            )
    return samples


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


def parse_test_labels(raw_Labels):
    """
    The format is stored in a 18d array (x,y,w,h,face_type,x1,y1,w1,h1, occ_type,
    occ_degree, gender, race, orientation, x2,y2,w2,h2), where
        (a) (x,y,w,h) is the bounding box of a face, 
        (b) face_type stands for the face type and has: 1 for masked face, 2 for
            unmasked face and 3 for invalid face.
        (c) (x1,y1,w1,h1) is the bounding box of the occluder. Note that (x1,y1)
            is related to the face bounding box position (x,y)
        (d) occ_type stands for the occluder type and has: 1 for simple, 2 for 
            complex and 3 for human body.
        (e) occ_degree stands for the number of occluded face parts
        (f) gender and race stand for the gender and race of one face
        (g) orientation stands for the face orientation/pose, and has: 1-left, 
            2-left frontal, 3-frontal, 4-right frontal, 5-right
        (h) (x2,y2,w2,h2) is the bounding box of the glasses and is set to 
            (-1,-1,-1,-1) when no glasses.  Note that (x2,y2) is related to the 
            face bounding box position (x,y)
    """
    return {
        'face': [raw_Labels[0], raw_Labels[1], raw_Labels[2], raw_Labels[3]],
        'face_type': raw_Labels[4],
        'occlude': {
            'location': [raw_Labels[5], raw_Labels[6], raw_Labels[7], raw_Labels[8]],
            'type': raw_Labels[9],
            'degree': raw_Labels[10]},
        'gender': raw_Labels[11],
        'race': raw_Labels[12],
        'orientation': raw_Labels[13],
        'glass': [raw_Labels[14], raw_Labels[15], raw_Labels[16], raw_Labels[17]]
    }


def draw_face(image, labels, color=(0, 255, 0)):
    x, y, w, h = labels['face']
    cv2.rectangle(image, (x, y, w, h), color, 2)


def draw_mask(image, labels, color=(0, 255, 0)):
    x, y, w, h = labels['face']
    _x, _y, _w, _h = labels['occlude']['location']
    cv2.rectangle(image, (x + _x, y + _y),
                  (x + _w, y + _h), color, 2)


def export_face(image, labels, export_file, target_size=112):
    # Crop the face
    x, y, w, h = labels['face']
    image_face = image[y:(y+h), x: (x+w)]

    # Resize and save image.
    image_face = cv2.resize(image_face, (target_size, target_size))
    cv2.imwrite(export_file, image_face)


if __name__ == "__main__":
    # Load annotations from the mat file.
    samples = load_labels(MAT_FILE, data_type='test')

    # loop through all the annotations and do processing. Here we are going to
    # extract all the occluded faces and save them in a new image file.
    for sample in tqdm(samples):
        labels = parse_test_labels(sample['lables'])
        image = cv2.imread(
            path.join(MAFA_ROOT, 'test/images', sample['image_file']))

        print(labels)
        draw_face(image, labels)
        draw_mask(image, labels)
        cv2.imshow('preview', image)
        if cv2.waitKey() == 27:
            break
        # face_image_file = path.join(
        #     EXPORT_DIR, 'images', sample['image_file'])
        # export_face(image, labels, face_image_file, 112)
