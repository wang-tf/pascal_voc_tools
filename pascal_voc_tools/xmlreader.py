#!/usr/bin/evn python3
# -*- coding:utf-8 -*-
"""
@File: decodeXml.py
@Author: ternencewang
@Desc: decode xml file of Pascal Voc annotation

"""
import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class XmlReader():
    """Decode the xml file"""
    def __init__(self, xml_file_path, debug=False):
        """
        Arguments:
            xml_file_path: str, the path of a xml file
        """
        self.debug = debug
        self.rewrite = False

        # assert input file path
        assert os.path.exists(xml_file_path), "ERROR: can not find file: {}".format(xml_file_path)
        self.file_path = xml_file_path

        # get root node of the xml
        self.tree = ET.parse(self.file_path)
        self.root = self.tree.getroot()

    def _save_element_info(self, save_dict, tag, text):
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

    def parse_element(self, element, save_dict=None):
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
                self._save_element_info(save_dict, child.tag, child.text)
            else:
                self._save_element_info(save_dict, child.tag, self.parse_element(child))

        return save_dict

    def load(self):
        """ Parse all annotations under the xml_root.
        """
        xml_root = self.root
        xml_info = self.parse_element(xml_root)
        return xml_info

    def __call__(self):
        """ run load()
        """
        return self.load()


class XmlTools(XmlReader):
    """ Smoe function for xml file.
    """
    def __init__(self, xml_file_path, debug=False):
        XmlReader.__init__(self, xml_file_path, debug)
        self.xml_info = self.load()

    def get_file_name(self):
        """Get file name node information in xml"""
        # filename = self.root.find('filename').text
        filename = self.xml_info['filename']
        return filename

    def get_width_and_height(self):
        """Get width and height of image in xml"""
        # for size in self.root.findall('size'):
        #     width = size.find('width').text
        #     height = size.find('height').text
        size = self.xml_info['size']
        width = size['width']
        height = size['height']
        return width, height

    def get_object_list(self):
        """Get all object name and number"""
        # obj_list = {}
        # for object in self.root.findall('object'):
        #     name = self.get_object_name(object)
        #     if name not in obj_list:
        #         obj_list[name] = 1
        #     else:
        #         obj_list[name] += 1
        obj_list = self.xml_info['object']
        return obj_list
