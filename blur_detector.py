"""A demonstration code of image blurness detection. The algorithm is from the 
technical paper:
Marichal, Xavier & Ma, Wei-Ying & Zhang, HongJiang. (1999). Blur determination 
in the compressed domain using DCT information. IEEE International Conference on
Image Processing. 2. 386 - 390 vol.2. 10.1109/ICIP.1999.822923. 
"""
import cv2
import numpy as np


class BlurDetector(object):

    def __init__(self):
        """Initialize a DCT based blur detector"""
        self.dct_threshold = 8.0
        self.max_hist = 0.1
        self.hist_weight = np.array([8, 7, 6, 5, 4, 3, 2, 1,
                                     7, 8, 7, 6, 5, 4, 3, 2,
                                     6, 7, 8, 7, 6, 5, 4, 3,
                                     5, 6, 7, 8, 7, 6, 5, 4,
                                     4, 5, 6, 7, 8, 7, 6, 5,
                                     3, 4, 5, 6, 7, 8, 7, 6,
                                     2, 3, 4, 5, 6, 7, 8, 7,
                                     1, 2, 3, 4, 5, 6, 7, 8
                                     ]).reshape(8, 8)
        self.weight_total = 344.0

    def check_image_size(self, image, block_size=8):
        """Make sure the image size is valid.
        Args:
            image: input image as a numpy array.
            block_size: the size of the minimal DCT block.
        Returns:
            result: boolean value indicating whether the image is valid.
            image: a modified valid image.
        """
        result = True
        height, width = image.shape[:2]
        _y = height % block_size
        _x = width % block_size

        pad_x = pad_y = 0

        if _y != 0:
            pad_y = block_size - _y
            result = False
        if _x != 0:
            pad_x = block_size - _x
            result = False

        image = cv2.copyMakeBorder(
            image, 0, pad_y, 0, pad_x, cv2.BORDER_REPLICATE)

        return result, image

    def get_blurness(self, image, block_size=8):
        """Estimate the blurness of an image.
        Args:
            image: image as a numpy array of shape [height, width, channels].
            block_size: the size of the minimal DCT block size.
        Returns:
            a float value represents the blurness.
        """
        # A 2D histogram.
        hist = np.zeros((8, 8), dtype=int)

        # Only the illumination is considered in blur.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Split the image into patches and do DCT on the image patch.
        height, width = image.shape
        round_v = int(height / block_size)
        round_h = int(width / block_size)
        for v in range(round_v):
            for h in range(round_h):
                v_start = v * block_size
                v_end = v_start + block_size
                h_start = h * block_size
                h_end = h_start + block_size

                image_patch = image[v_start:v_end, h_start:h_end]
                image_patch = np.float32(image_patch)
                patch_spectrum = cv2.dct(image_patch)
                patch_none_zero = np.abs(patch_spectrum) > self.dct_threshold
                hist += patch_none_zero.astype(int)

        _blur = hist < self.max_hist * hist[0, 0]
        _blur = (np.multiply(_blur.astype(int), self.hist_weight)).sum()
        return _blur/self.weight_total


if __name__ == "__main__":
    bd = BlurDetector()
    image = cv2.imread('/home/robin/Desktop/face.jpg')
    result, image = bd.check_image_size(image)
    if not result:
        print("Image expanded.")
    blur = bd.get_blurness(image)
    print("Blurness: {:.2f}".format(blur))
