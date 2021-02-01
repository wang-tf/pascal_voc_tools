import os
import glob
import numpy as np
import cv2

from _xml_parser import XmlParser


class FalsePositiveArgument:
    """
    """
    def __init__(self, fp_image_list, voc_root_dir, save_dir):
        """
        """
        self.fp_image_list = fp_image_list
        self.voc_root_dir = voc_root_dir
        self.save_dir = save_dir

        self.images_dir = os.path.join(voc_root_dir, 'JPEGImages')
        self.xmls_dir = os.path.join(voc_root_dir, 'Annotations')

    def image_augment(self, image):
        """增强贴图

        Args:
            image: ndarray, an opencv image;
        Returns:
            an augmented image.
        """
        image_augmented = image
        return image_augmented

    def load_backgroud_images(self):
        """
        """
        images = glob.glob(os.path.join(self.images_dir, '*.jpg'))
        return images

    def load_annotation(self, image_id):
        """倒入一个xml数据

        Arguments:
            image_id: str, an image id;
        Returns:
            the xml data about input image id.
        Raises:
            AssertError: can not find corresponding xml file.
        """
        xml_path = os.path.join(self.xmls_dir, image_id + '.xml')
        assert os.path.exists(xml_path), xml_path

        parser = XmlParser()
        xml_data = parser.load(xml_path)
        return xml_data

    def get_used_mask(self, image, xml_data):
        """根据xml数据，产生一个mask，标注有目标的位置

        Arguments:
            image: ndarray, an opencv image;
            xml_data: dict, xml data;
        Returns:
            image_mask
        """
        mask = np.zeros((image.shape[0], image.shape[1]))

        for obj in xml_data['object']:
            xmin = int(obj['xmin'])
            ymin = int(obj['ymin'])
            xmax = int(obj['xmax'])
            ymax = int(obj['ymax'])

            mask[ymin:ymax, xmin:xmax] = 1
        return mask

    def ramdom_pad(self, image_path, fp_image, try_time=10):
        """try to pad fp_image to image.
        """
        assert os.path.exists(image_path), image_path

        image = cv2.imread(image_path)
        image_id = os.path.basename(image_path).split('.')[0]
        xml_data = self.load_annotation(image_id)
        mask = self.get_used_mask(image, xml_data)

        for try_index in range(try_time):
            # TODO: 随机选择一个mask为0的位置，判断以该位置为左上角点是否可行
            pass
