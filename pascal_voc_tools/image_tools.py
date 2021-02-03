# coding:utf-8
import glob
import logging
import os

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageWrapper(object):
    """Some image tools.
    """
    def __init__(self):
        self.path = None
        self.data = None

        self.width = None
        self.height = None
        self.depth = None

    def load(self, image_path: str):
        """load image data.

        Arguments:
            image_path: image file path.
        """
        assert os.path.exists(image_path), image_path
        self.data = cv2.imread(image_path)
        self.path = image_path
        self.height = self.data.shape[0]
        self.width = self.data.shape[1]
        self.depth = self.data.shape[2] if len(self.data.shape) == 3 else 1
        return self

    def resize_by_rate(self, rate: float):
        """Resize image.

        Arguments:
            rate: image resize rate.
        """
        self.data = cv2.resize(self.data, None, fx=rate, fy=rate)
        self.height = self.data.shape[0]
        self.width = self.data.shape[1]
        return self

    def resize_letter_box(self, width: int, height: int):
        """using letter box to resize image data.

        Arguments:
            width: new image width.
            height: new image height.

        Returns:
            a flot of resize rate.
        """
        if self.depth == 1:
            mask_image = np.zeros((height, width), dtype=np.uint8)
        else:
            mask_image = np.zeros((height, width, self.depth), dtype=np.uint8)

        rate = min(float(width) / self.width, float(height) / self.height)
        new_width = int(self.width * rate)
        new_height = int(self.height * rate)

        horizion_bias = int((width - new_width) / 2)
        vertical_bias = int((height - new_height) / 2)

        resized_image = cv2.resize(self.data, (new_width, new_height))
        mask_image[vertical_bias:vertical_bias + new_height,
                   horizion_bias:horizion_bias + new_width] = resized_image

        self.data = mask_image
        self.height = new_height
        self.width = new_width
        return rate, (vertical_bias, horizion_bias)

    def save(self, save_path):
        """save image data to a file.

        Arguments:
            save_path: a image file path.
        """
        cv2.imwrite(save_path, self.data)
        return self

    def crop_image(self, split_bboxes):
        """Split an image to some subimages.

        Arguments:
            split_bboxes: list, like[[xmin, ymin, xmax, ymax], ]

        Returns:
            images: list, all subimages.
        """
        subimages = []
        for bbox in split_bboxes:
            sub_image = ImageWrapper()
            sub_image.data = self.data[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            sub_image.width = bbox[2] - bbox[0]
            sub_image.height = bbox[3] - bbox[1]
            sub_image.depth = self.depth
            subimages.append(sub_image)
        return subimages


def verify_image(jpeg_path):
    """Verify image format.
    Args:
        jpeg_path: str, the image path.

    Returns:
        result: bool, if image ok, True will return.
    """
    assert os.path.exists(jpeg_path), jpeg_path
    try:
        Image.open(jpeg_path).verify()
    except Exception as err:
        logger.exception(err)
        return False
    return True


def image_convert_2_jpg(images_dir):
    """
    Change image format from others to jpg.
    Args:
        jpeg_dir: str, the dir only have image.
    """
    image_path_list = glob.glob(os.path.join(images_dir, '*.*'))
    for jpeg_path in image_path_list:
        suffix = os.path.basename(jpeg_path).split('.')[-1]
        if suffix != 'jpg' and verify_image(jpeg_path):
            image = Image.open(jpeg_path)
            save_path = jpeg_path.replace('.{}'.format(suffix), '.jpg')
            image.save(save_path)
            os.remove(jpeg_path)
            print('Image from {} to {}'.format(jpeg_path, save_path))

    return True
