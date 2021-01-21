# -*- coding:utf-8 -*-
"""
Resize the whole dataset including images and xmls
"""

import os
import glob
import cv2
import shutil
import tqdm

from ._xml_parser import XmlParser
from .tools import resize_image_by_size


class DatasetResize():
    """Resize voc dataset.

    Attributes:
        root_voc_dir: str, the path of pascal voc dir include VOC2007.
        save_root_dir: str, the path of save path, default path is input path.
    """
    def __init__(self, root_voc_dir, save_root_dir=None):
        """set save path
        """
        self.root_dir = root_voc_dir
        self.save_root_dir = self.root_dir if save_root_dir is None else save_root_dir

        self.annotations_dir = os.path.join(self.root_dir, 'Annotations')
        assert os.path.exists(self.annotations_dir), self.annotations_dir

        self.images_dir = os.path.join(self.root_dir, 'JPEGImages')
        assert os.path.exists(self.images_dir), self.images_dir

        # set save path
        self.save_annotations_dir = os.path.join(self.save_root_dir,
                                                 'Annotations')
        self.save_images_dir = os.path.join(self.save_root_dir, 'JPEGImages')
        if not os.path.exists(self.save_annotations_dir):
            os.makedirs(self.save_annotations_dir)
        if not os.path.exists(self.save_images_dir):
            os.makedirs(self.save_images_dir)

    def _get_annotations(self):
        annotations_file_list = glob.glob(
            os.path.join(self.annotations_dir, '*.xml'))
        return annotations_file_list

    def resize_tuple_by_rate(self,
                             rate,
                             image_path,
                             xml_path,
                             save_image_path=None,
                             save_xml_path=None,
                             min_obj_size=8):
        """Resize a image and coresponding xml by rate

        Args:
            rate: float or int, scale size.
            image_path: str, the path of image.
            xml_path: str, the path of xml file.
            save_image_path: str, image save path, default is image_path.
            save_xml_path: str, xml save path, default is xml_path.
        """
        assert os.path.exists(image_path), image_path
        assert os.path.exists(xml_path), xml_path

        save_image_path = image_path if save_image_path is None else save_image_path
        save_xml_path = xml_path if save_xml_path is None else save_xml_path

        # resize image and save
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, None, fx=rate, fy=rate)
        cv2.imwrite(save_image_path, image_resized)

        # resize annotation and save
        parser = XmlParser()
        xml_data = parser.load(xml_path)

        original_width = int(xml_data['size']['width'])
        original_height = int(xml_data['size']['height'])

        new_width = int(original_width * rate)
        new_height = int(original_height * rate)

        horizion_bias = 0
        vertical_bias = 0

        xml_data['size']['width'] = new_width
        xml_data['size']['height'] = new_height

        new_objs = []
        for index, obj in enumerate(xml_data['object']):
            xmin = int(float(obj['bndbox']['xmin']))
            ymin = int(float(obj['bndbox']['ymin']))
            xmax = int(float(obj['bndbox']['xmax']))
            ymax = int(float(obj['bndbox']['ymax']))

            new_bndbox = self.resize_bbox(rate, horizion_bias, vertical_bias,
                                          [xmin, ymin, xmax, ymax])
            new_bndbox = list(map(int, new_bndbox))
            xmin, ymin, xmax, ymax = new_bndbox
            if xmax - xmin < min_obj_size or ymax - ymin < min_obj_size:
                print(f'The new size of {xmin}, {ymin}, {xmax}, {ymax}',
                      f' is smaler than {min_obj_size}*{min_obj_size}. delete')
                continue
            xml_data['object'][index]['bndbox']['xmin'] = str(xmin)
            xml_data['object'][index]['bndbox']['ymin'] = str(ymin)
            xml_data['object'][index]['bndbox']['xmax'] = str(xmax)
            xml_data['object'][index]['bndbox']['ymax'] = str(ymax)
            new_objs.append(xml_data['object'][index])
        xml_data['object'] = new_objs

        save_xml_path = xml_path if save_xml_path is None else save_xml_path
        parser.save(save_xml_path, xml_data)
        return 1

    def resize_tuple_by_min_size(self,
                                 min_size,
                                 image_path,
                                 xml_path,
                                 save_image_path=None,
                                 save_xml_path=None):
        """Resize a image and coresponding xml by set min size and keep aspect ratio

        Args:
            min_size: int, new image min size;
            image_path: str, image path;
            xml_path: str, xml file about image;
            save_image_path: str, save image path;
            save_xml_path: str, save xml path.
        """
        assert os.path.exists(image_path), image_path
        assert os.path.exists(xml_path), xml_path

        save_image_path = image_path if save_image_path is None else save_image_path
        save_xml_path = xml_path if save_xml_path is None else save_xml_path

        image = cv2.imread(image_path)
        height, width = image.shape[0:2]
        rate = min_size / min(height, width)
        new_height = int(height * rate)
        new_width = int(width * rate)

        self.resize_tuple_by_rate(rate, image_path, xml_path, save_image_path,
                                  save_xml_path)

    def resize_tuple_by_size(self,
                             width,
                             height,
                             image_path,
                             xml_path,
                             save_image_path=None,
                             save_xml_path=None,
                             min_obj_size=8):
        """Resize a image and coresponding xml by set fix size

        Args:
            width: int, new image width;
            height: int, new image height;
            image_path: str, image path;
            xml_path: str, xml file about image;
            save_image_path: str, save image path;
            save_xml_path: str, save xml path.
        """
        assert os.path.exists(image_path), image_path
        assert os.path.exists(xml_path), xml_path

        save_image_path = image_path if save_image_path is None else save_image_path
        save_xml_path = xml_path if save_xml_path is None else save_xml_path

        self.resize_image_by_size(image_path, width, height, save_image_path)
        self.resize_xml_by_size(xml_path,
                                width,
                                height,
                                save_xml_path,
                                min_obj_size=min_obj_size)
        return

    def resize_image_by_size(self,
                             image_path,
                             width,
                             height,
                             save_image_path=None):
        """Resize an image

        Args:
            image_path: str, image path;
            width: int, new image width;
            height: intn new image height;
            save_image_path: image save path.
        """
        assert os.path.exists(image_path), image_path

        image = cv2.imread(image_path)
        resized_image, rate, bias = resize_image_by_size(image,
                                                         width=width,
                                                         height=height)

        cv2.imwrite(save_image_path, resized_image)
        assert os.path.exists(
            save_image_path), 'Result was not saved: '.format(save_image_path)
        return

    def resize_bbox(self, rate, horizion_bias, vertical_bias, bbox):
        """resize a bbox

        Args:
            rate: box change rate;
            horizion_bias: xmin and xmax will add it;
            vertical_bias: ymin and ymax will add it,
            bbox: list, [xmin, ymin, xmax, ymax].
        Returns:
            new bbox list
        """
        xmin, ymin, xmax, ymax = bbox
        xmin = xmin * rate + horizion_bias
        ymin = ymin * rate + vertical_bias
        xmax = xmax * rate + horizion_bias
        ymax = ymax * rate + vertical_bias
        return [xmin, ymin, xmax, ymax]

    def resize_xml_by_size(self,
                           xml_path,
                           width,
                           height,
                           save_xml_path=None,
                           min_obj_size=8):
        """
        Args:
            xml_path: str, xml file path;
            width: int, new image width;
            height: int, new image height;
            save_xml_path: str, save path;
            min_obj_size: int, if resized bbox smaller than it will be pass.
        """
        parser = XmlParser()
        xml_data = parser.load(xml_path)

        original_width = int(xml_data['size']['width'])
        original_height = int(xml_data['size']['height'])

        rate = min(
            float(width) / original_width,
            float(height) / original_height)
        new_width = int(original_width * rate)
        new_height = int(original_height * rate)

        horizion_bias = int((width - new_width) / 2)
        vertical_bias = int((height - new_height) / 2)

        xml_data['size']['width'] = width
        xml_data['size']['height'] = height

        new_objs = []
        for index, obj in enumerate(xml_data['object']):
            xmin = int(float(obj['bndbox']['xmin']))
            ymin = int(float(obj['bndbox']['ymin']))
            xmax = int(float(obj['bndbox']['xmax']))
            ymax = int(float(obj['bndbox']['ymax']))

            new_bndbox = self.resize_bbox(rate, horizion_bias, vertical_bias,
                                          [xmin, ymin, xmax, ymax])
            new_bndbox = list(map(int, new_bndbox))
            xmin, ymin, xmax, ymax = new_bndbox
            if xmax - xmin < min_obj_size or ymax - ymin < min_obj_size:
                print(
                    f'The new size of {xmin}, {ymin}, {xmax}, {ymax} is smaler than {min_obj_size}*{min_obj_size}. delete'
                )
                continue
            xml_data['object'][index]['bndbox']['xmin'] = str(xmin)
            xml_data['object'][index]['bndbox']['ymin'] = str(ymin)
            xml_data['object'][index]['bndbox']['xmax'] = str(xmax)
            xml_data['object'][index]['bndbox']['ymax'] = str(ymax)
            new_objs.append(xml_data['object'][index])
        xml_data['object'] = new_objs

        save_xml_path = xml_path if save_xml_path is None else save_xml_path
        parser.save(save_xml_path, xml_data)
        return

    def resize_by_rate(self, rate, min_obj_size=8):
        """Resize the whole dataset

        Args:
            rate: float or int, scale size.
        """
        annotations_file_list = self._get_annotations()

        print('Resizing dataset ...')
        for xml_path in tqdm.tqdm(annotations_file_list):
            image_path = self.get_image_path_by_xml_path(xml_path)
            save_xml_path = self.get_save_path(xml_path)
            save_image_path = self.get_save_path(image_path)

            self.resize_tuple_by_rate(rate,
                                      image_path,
                                      xml_path,
                                      save_image_path,
                                      save_xml_path,
                                      min_obj_size=min_obj_size)

    def resize_by_min_size(self, min_size):
        """Resize whole dataset by set a min image size.

        Args:
            min_size: int, new image min size.
        """
        annotations_file_list = self._get_annotations()

        print('Resizing dataset ...')
        for xml_path in tqdm.tqdm(annotations_file_list):
            image_path = self.get_image_path_by_xml_path(xml_path)
            save_xml_path = self.get_save_path(xml_path)
            save_image_path = self.get_save_path(image_path)

            self.resize_tuple_by_min_size(min_size, image_path, xml_path,
                                          save_image_path, save_xml_path)

    def resize_by_size(self, width, height, min_obj_size=8):
        """Resize whole dataset by set a fix image size.

        Args:
            width: int, new image width;
            height: intn new image height.
        """
        annotations_file_list = self._get_annotations()

        print('Resizing dataset ...')
        for xml_path in tqdm.tqdm(annotations_file_list):
            image_path = self.get_image_path_by_xml_path(xml_path)
            save_xml_path = self.get_save_path(xml_path)
            save_image_path = self.get_save_path(image_path)

            self.resize_tuple_by_size(width,
                                      height,
                                      image_path,
                                      xml_path,
                                      save_image_path,
                                      save_xml_path,
                                      min_obj_size=min_obj_size)
        return

    def get_save_path(self, path):
        """return new xml save path

        Args:
            path: str, xml path.

        Returns:
            new xml path
        """
        name = os.path.basename(path)
        save_dir = self.save_annotations_dir if name[
            -4:] == '.xml' else self.save_images_dir
        save_path = os.path.join(save_dir, name)
        return save_path

    def get_image_path_by_xml_path(self, xml_path):
        """using xml path to inference image path

        Args:
            xml_path: str, xml file path.
        
        Returns:
            image path in current dataset.
        """
        xml_name = os.path.basename(xml_path)
        image_path = os.path.join(self.images_dir,
                                  xml_name.replace('.xml', '.jpg'))
        return image_path

    def copy_imagesets(self, imagesets_dir=None):
        """Copy some text file in Main dir from root dir to save dir

        Args:
            images_dir: str, the path of ImageSets/Main.
        """
        if imagesets_dir is None:
            imagesets_dir = os.path.join(self.root_dir, 'ImageSets/Main')

        # make save dir
        save_dir = os.path.join(self.save_root_dir, 'ImageSets/Main')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for file_name in ['train.txt', 'val.txt', 'trainval.txt', 'test.txt']:
            file_path = os.path.join(imagesets_dir, file_name)
            if os.path.exists(file_path):
                shutil.copy2(file_path, save_dir)
            else:
                print('Can not find path: {}'.format(file_path))
