# -*- coding:utf-8 -*-

import os
import numpy as np
import cv2
import tqdm
import random
from ._xml_parser import XmlParser


def pad_image(bg_image, pad_image, top_left):
    """pad a small image into a background image.

    Args:
        bg_image: ndarray, an image;
        pad_image: ndarray, an image;
        top_left: list, pad start point.
    Returns:
        masked image.
    Raise:
        AssertError: the pad size is not smaller than bg_image.
    """
    mask_image = bg_image.copy()
    # make sure the mask image is bigger than pad image.
    pad_image_height, pad_image_width = pad_image.shape[:2]
    assert mask_image.shape[0] >= top_left[0] + pad_image_height
    assert mask_image.shape[1] >= top_left[1] + pad_image_width

    mask_image[top_left[0]:top_left[0] + pad_image_height,
               top_left[1]:top_left[1] + pad_image_width] = pad_image
    return mask_image


def pad_object(bg_object, pad_obj, top_left):
    """
    """
    for obj in pad_obj:
        obj['bndbox']['xmin'] = str(int(obj['bndbox']['xmin']) + top_left[1])
        obj['bndbox']['ymin'] = str(int(obj['bndbox']['ymin']) + top_left[0])
        obj['bndbox']['xmax'] = str(int(obj['bndbox']['xmax']) + top_left[1])
        obj['bndbox']['ymax'] = str(int(obj['bndbox']['ymax']) + top_left[0])
    return bg_object + pad_obj


def pad2_horizon(image1, image2, image1_object, image2_object):
    """ Horizon concat two images.
    """
    bg_image = np.zeros(
        (max(image1.shape[0],
             image2.shape[0]), image1.shape[1] + image2.shape[1], 3),
        dtype=np.uint8)
    bg_image = pad_image(bg_image, image1, [0, 0])
    bg_image = pad_image(bg_image, image2, [0, image1.shape[1]])

    bg_annotation_object = pad_object(image1_object, image2_object,
                                      [0, image1.shape[1]])
    return bg_image, bg_annotation_object


def pad2_vertical(image1, image2, image1_object, image2_object):
    """ Vertical concat two images.
    """
    bg_image = np.zeros((image1.shape[0] + image2.shape[0],
                         max(image1.shape[1], image2.shape[1]), 3),
                        dtype=np.uint8)
    bg_image = pad_image(bg_image, image1, [0, 0])
    bg_image = pad_image(bg_image, image2, [image1.shape[0], 0])

    bg_annotation_object = pad_object(image1_object, image2_object,
                                      [image1.shape[0], 0])
    return bg_image, bg_annotation_object


def pad4(images_list, objects_list):
    """ 2*2 concat four images.
    """
    assert len(images_list) == 4
    assert len(objects_list) == 4

    bg_image1, bg_object1 = pad2_horizon(images_list[0], images_list[1],
                                         objects_list[0], objects_list[1])
    bg_image2, bg_object2 = pad2_horizon(images_list[2], images_list[3],
                                         objects_list[2], objects_list[3])
    bg_image, bg_object = pad2_vertical(bg_image1, bg_image2, bg_object1,
                                        bg_object2)
    return bg_image, bg_object


class PadDataset():
    """ Concat 2 or 4 images to one.

    Attributes:
        voc_root: str, PascalVOC foramt dataset;
        voc_save_root: str, save path;
        main_name: str, data set name.
    """
    def __init__(self, voc_root, voc_save_root, main_name='train'):
        self.voc_root = voc_root
        self.main_name = main_name
        self.jpeg_dir = os.path.join(voc_root, 'JPEGImages')
        self.annotation_dir = os.path.join(voc_root, 'Annotations')
        self.main_path = os.path.join(voc_root, 'ImageSets/Main',
                                      main_name + '.txt')

        assert os.path.exists(self.jpeg_dir)
        assert os.path.exists(self.annotation_dir)
        assert os.path.exists(self.main_path)

        self.save_jpeg_dir = os.path.join(voc_save_root, 'JPEGImages')
        self.save_annotation_dir = os.path.join(voc_save_root, 'Annotations')
        self.save_main_dir = os.path.join(voc_save_root, 'ImageSets/Main')
        if not os.path.exists(self.save_jpeg_dir):
            os.makedirs(self.save_jpeg_dir)
        if not os.path.exists(self.save_annotation_dir):
            os.makedirs(self.save_annotation_dir)
        if not os.path.exists(self.save_main_dir):
            os.makedirs(self.save_main_dir)

        with open(self.main_path) as f:
            names = f.read().strip().split('\n')
        self.ids = [n.strip() for n in names]
        print('Find {} data ids.'.format(len(self.ids)))

    def random_pad4(self):
        """ Concat 2*2 images.
        """
        random.shuffle(self.ids)
        save_ids = []
        for i in tqdm.tqdm(range(len(self.ids) // 4)):
            images = []
            annotation_objects = []
            for j in range(4):
                image_id = self.ids[i * 4 + j]
                image = cv2.imread(
                    os.path.join(self.jpeg_dir, image_id + '.jpg'))
                images.append(image)

                image_ann = XmlParser().load(
                    os.path.join(self.annotation_dir, image_id + '.xml'))
                if 'object' not in image_ann:
                    image_ann['object'] = []
                image_obj = image_ann['object']
                annotation_objects.append(image_obj)

            pad_image, pad_object = pad4(images, annotation_objects)
            save_id = '+'.join(self.ids[i * 4:i * 4 + 4])
            image_save_path = os.path.join(self.save_jpeg_dir,
                                           save_id + '.jpg')
            cv2.imwrite(image_save_path, pad_image)
            save_parser = XmlParser()
            save_parser.set_head(path=image_save_path,
                                 height=pad_image.shape[0],
                                 width=pad_image.shape[1])
            save_parser.template_parameters['object'] = pad_object
            save_parser.save(
                os.path.join(self.save_annotation_dir, save_id + '.xml'))
            save_ids.append(save_id)

        with open(os.path.join(self.save_main_dir, self.main_name + '.txt'),
                  'w') as f:
            f.write('\n'.join(save_ids))

    def random_pad2_horizon(self):
        """ Horizon concat 2 images.
        """
        random.shuffle(self.ids)
        save_ids = []
        for i in tqdm.tqdm(range(len(self.ids) // 2)):
            images = []
            annotation_objects = []
            for j in range(2):
                image_id = self.ids[i + j]
                image = cv2.imread(
                    os.path.join(self.jpeg_dir, image_id + '.jpg'))
                images.append(image)

                image_ann = XmlParser().load(
                    os.path.join(self.annotation_dir, image_id + '.xml'))
                if 'object' not in image_ann:
                    image_ann['object'] = []
                image_obj = image_ann['object']
                annotation_objects.append(image_obj)

            pad_image, pad_object = pad2_horizon(images[0], images[1],
                                                 annotation_objects[0],
                                                 annotation_objects[1])
            save_id = '+'.join(self.ids[i * 2:i * 2 + 2])
            image_save_path = os.path.join(self.save_jpeg_dir,
                                           save_id + '.jpg')
            cv2.imwrite(image_save_path, pad_image)
            save_parser = XmlParser()
            save_parser.set_head(path=image_save_path,
                                 height=pad_image.shape[0],
                                 width=pad_image.shape[1])
            save_parser.template_parameters['object'] = pad_object
            save_parser.save(
                os.path.join(self.save_annotation_dir, save_id + '.xml'))
            save_ids.append(save_id)

        with open(os.path.join(self.save_main_dir, self.main_name + '.txt'),
                  'w') as f:
            f.write('\n'.join(save_ids))

    def random_pad2_vertical(self):
        """ Vertical concat 2 images.
        """
        random.shuffle(self.ids)
        save_ids = []
        for i in tqdm.tqdm(range(len(self.ids) // 2)):
            images = []
            annotation_objects = []
            for j in range(2):
                image_id = self.ids[i + j]
                image = cv2.imread(
                    os.path.join(self.jpeg_dir, image_id + '.jpg'))
                images.append(image)

                image_ann = XmlParser().load(
                    os.path.join(self.annotation_dir, image_id + '.xml'))
                if 'object' not in image_ann:
                    image_ann['object'] = []
                image_obj = image_ann['object']
                annotation_objects.append(image_obj)

            pad_image, pad_object = pad2_vertical(images[0], images[1],
                                                  annotation_objects[0],
                                                  annotation_objects[1])
            save_id = '+'.join(self.ids[i * 2:i * 2 + 2])
            image_save_path = os.path.join(self.save_jpeg_dir,
                                           save_id + '.jpg')
            cv2.imwrite(image_save_path, pad_image)
            save_parser = XmlParser()
            save_parser.set_head(path=image_save_path,
                                 height=pad_image.shape[0],
                                 width=pad_image.shape[1])
            save_parser.template_parameters['object'] = pad_object
            save_parser.save(
                os.path.join(self.save_annotation_dir, save_id + '.xml'))
            save_ids.append(save_id)

        with open(os.path.join(self.save_main_dir, self.main_name + '.txt'),
                  'w') as f:
            f.write('\n'.join(save_ids))
