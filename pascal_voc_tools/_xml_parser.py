#!/usr/bin/evn python
# -*- coding:utf-8 -*-
"""
@File: decodeXml.py
@Author: ternencewang
@Desc: decode xml file of Pascal Voc annotation
       write a xml file about pascal voc annotation.
<annotation>
    <folder>{{ folder }}</folder>
    <filename>{{ filename }}</filename>
    <path>{{ path }}</path>
    <source>
        <database>{{ database }}</database>
    </source>
    <size>
        <width>{{ size.width }}</width>
        <height>{{ size.height }}</height>
        <depth>{{ size.depth }}</depth>
    </size>
    <segmented>{{ segmented }}</segmented>{% for obj in object %}
    <object>
        <name>{{ obj.name }}</name>
        <pose>{{ obj.pose }}</pose>
        <truncated>{{ obj.truncated }}</truncated>
        <difficult>{{ obj.difficult }}</difficult>
        <bndbox>
            <xmin>{{ obj.bndbox.xmin }}</xmin>
            <ymin>{{ obj.bndbox.ymin }}</ymin>
            <xmax>{{ obj.bndbox.xmax }}</xmax>
            <ymax>{{ obj.bndbox.ymax }}</ymax>
        </bndbox>
    </object>{% endfor %}
</annotation>

"""
import os
from jinja2 import Environment, PackageLoader
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from PIL import Image


class XmlParser(object):
    """Decode the xml file"""

    def __init__(self):
        self.file_path = None
        self.data = None
        self.annotation_template = None
        self.template_parameters = None

        # some peramaters
        self.size = []

    def load(self, xml_file_path):
        """
        Arguments:
        ==========
            xml_file_path: str, the path of a xml file
        """
        assert os.path.exists(xml_file_path), xml_file_path

        self.file_path = xml_file_path
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        self.data = self.parse_element(root)
        return self.data

    def _save_element_info(self, save_dict, tag, text):
        """Save tag and text to save_dict.
        if tag not in save_dict, it will return like save_dict[tag] = text,
        otherwise it will like save_dict[tag] = [text, ]

        Arguments:
        ==========
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
        ==========
            element: element, an element type node in xml tree.
            save_dict: dict, save result.
        Returns:
        ========
            save_dict: dict, like {'path': './', 'segmented': '0', 'object': [{}]}.
        """
        if save_dict is None:
            save_dict = {}

        for child in element:
            if len(child) == 0:
                self._save_element_info(save_dict, child.tag, child.text)
            else:
                self._save_element_info(
                    save_dict, child.tag, self.parse_element(child))

        return save_dict

    def set_data_head(self, path='', width=0, height=0, depth=3, database='Unknown', segmented=0):
        """Generate a xml file

        Arguments:
        ==========
            path: str, arg in xml about image.
            width: int or str, image width.
            height: int or str, image height.
            depth: int or str, image channle.
            database: str, default='Unknown'.
            segmented: int or str, default=0.
        """
        environment = Environment(loader=PackageLoader(
            'pascal_voc_tools', 'templates'), keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'size': {'width': width, 'height': height, 'depth': depth, },
            'source': {'database': database, },
            'segmented': segmented,
            'object': []
        }

    def add_object(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        """add an object info

        Arguments:
        ==========
            name: str, class name.
            xmin: int, left.
            ymin: int, top.
            xmax: int, right.
            ymax: int, bottom.
            pose: str, default is 'Unspecified'.
            truncated: str, default is 0.
            difficult: int, default is 0.
        """
        self.template_parameters['object'].append({
            'name': name,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
            'bndbox': {
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
            },
        })

    def save(self, annotation_path, image_parameters=None):
        """Write a xml file to save info.
        Arguments:
        ==========
            annotation_path: str, the path of xml to save.
        """
        if image_parameters is not None:
            for k, v in image_parameters.items():
                self.template_parameters[k] = image_parameters[k]

        with open(annotation_path, 'w') as xml_file:
            content = self.annotation_template.render(
                **self.template_parameters)
            xml_file.write(content)
