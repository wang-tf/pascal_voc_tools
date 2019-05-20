#!/usr/bin/env python3

import os
import glob
from PIL import Image
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from xmlwriter import XmlWriter


def verify_image(jpeg_path):
    """Verify image format.
    Arguments:
    =========
        jpeg_path: str, the image path.
    Return:
    ======
        result: bool, if image ok, True will return.
    """
    assert os.path.exists(jpeg_path), jpeg_path
    try:
        Image.open(jpeg_path).verify()
    except:
        return False
    return True


def _save_element_info(save_dict, tag, text):
    """Save tag and text to save_dict.
    if tag not in save_dict, it will return like save_dict[tag] = text,
    otherwise it will like save_dict[tag] = [text, ]
    Arguments:
        save_dict: dict, to save.
        tag: str, the key to save.
        text: str, the value to save.
    """
    if tag not in save_dict:
        if tag != 'object':
            save_dict[tag] = text
        else:
            save_dict[tag] = [text]
    else:
        if not isinstance(save_dict[tag], list):
            save_dict[tag] = [save_dict[tag]]
        save_dict[tag].append(text)


def _parse_element(element, save_dict=None):
    """Parse all information in element and save to save_dict.
    Arguments:
        element: element, an element type node in xml tree.
        save_dict: dict, save result.
    Returns:
        save_dict: dict, like {'path': './', 'segmented': '0', 'object': [{}]}.
    """
    if save_dict is None:
        save_dict = {}

    for child in element:
        if len(child) == 0:
            _save_element_info(save_dict, child.tag, child.text)
        else:
            _save_element_info(save_dict, child.tag, _parse_element(child))

    return save_dict


def xmlread(file_path, debug=False):
    """
    Parse all information in element and save to save_dict.
    Arguments:
    =========
        file_path: str, the path of a xml file.
    Returns:
    =======
        save_dict: dict, like {'path': './', 'segmented': '0', 'object': [{}]}.
    """
    # assert input file path
    assert os.path.exists(file_path), "ERROR: can not find file: {}".format(file_path)

    # get root node of the xml
    tree = ET.parse(file_path)
    root = tree.getroot()

    xml_info = _parse_element(root)
    return xml_info


def xmlwrite(save_path, xml_dict):
    """
    Save formated xml_dict to a xml file.
    Arguments:
    =========
        save_path: str, the xml path for saving;
        xml_dict: dict, original data.
    """
    writer = XmlWriter()
    writer.save(save_path, xml_dict)


def check_jpg_xml_match(xml_dir, jpeg_dir):
    """
    Check matching degree about xml files and jpeg files.
    Arguments:
    =========
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
    Arguments:
    =========
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
