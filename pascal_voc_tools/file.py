# -*- coding:utf-8 -*-

import os
import glob
from PIL import Image
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


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
    except:
        return False
    return True


def check_jpg_xml_match(xml_dir, jpeg_dir):
    """
    Check matching degree about xml files and jpeg files.
    Args:
        xml_dir: str, the dir including xml files;
        jpeg_dir: str, the dir including jpeg files.
    """
    # arguemnts check
    assert os.path.exists(xml_dir), xml_dir
    assert os.path.exists(jpeg_dir), jpeg_dir

    # get name list
    xml_file_list = glob.glob(os.path.join(xml_dir, '*.xml'))
    jpeg_file_list = glob.glob(os.path.join(jpeg_dir, '*.jpg'))
    xml_name_list = [os.path.basename(path).split('.')[0] for path in xml_file_list]
    jpeg_name_list = [os.path.basename(path).split('.')[0] for path in jpeg_file_list]

    inter = list(set(xml_name_list).intersection(set(jpeg_name_list)))
    xml_diff = list(set(xml_name_list).difference(set(jpeg_name_list)))
    jpeg_diff = list(set(jpeg_name_list).difference(set(xml_name_list)))

    # print result and return matched list
    print('Find {} xml, {} jpg, matched {}.'.format(len(xml_file_list), len(jpeg_file_list), len(inter)))
    if len(xml_diff):
        print("Only have xml file: {}\n{}".format(len(xml_diff), xml_diff))
    if len(jpeg_diff):
        print("Only have jpg file: {}\n{}".format(len(jpeg_diff), jpeg_diff))

    return inter


def check_image_format(jpeg_dir):
    """
    Change image format from others to jpg.
    Args:
        jpeg_dir: str, the dir only have image.
    """
    image_path_list = glob.glob(os.path.join(jpeg_dir, '*.*'))
    for jpeg_path in image_path_list:
        suffix = os.path.basename(jpeg_path).split('.')[-1]
        if suffix != 'jpg' and verify_image(jpeg_path):
            image = Image.open(jpeg_path)
            save_path = jpeg_path.replace('.{}'.format(suffix), '.jpg')
            image.save(save_path)
            os.remove(jpeg_path)
            print('Image from {} to {}'.format(jpeg_path, save_path))

    return True


def check_xml_info(xml_info):
    """
    Args:
        xml_info: dict, including xml data. 
    """
    # check image path
    assert 'path' in xml_info, 'Can not find key(path).'

    image_path = xml_info['path']

    assert os.path.exists(image_path), image_path
    image = Image.open(image_path)
    
    # check image size
    assert 'size' in xml_info, 'Can not find key(size).'
    size = xml_info['size']
    width, height = image.size()
    assert int(size['width']) == width
    assert int(size['height']) == height
    depth = 1 if len(image.mode()) == 1 else 3
    assert int(size['depth']) == depth

    # check object
    objects = xml_info['object']
    for one_obj in objects:
        bndbox = one_obj['bndbox']
        xmin = int(bndbox['xmin'])
        ymin = int(bndbox['ymin'])
        xmax = int(bndbox['xmax'])
        ymax = int(bndbox['ymax'])
        assert xmin > 0, xmin
        assert ymin > 0, ymin
        assert xmax < width, xmax
        assert ymax < height, ymax
