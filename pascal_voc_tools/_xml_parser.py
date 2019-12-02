# -*- coding:utf-8 -*-
"""
decode xml file of Pascal Voc annotation write a xml file about pascal voc annotation.
"""
import os
from jinja2 import Environment, PackageLoader
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class XmlParser():
    """Code and Decode the xml file"""

    def __init__(self):
        self.file_path = None

        self.template_parameters = {}
        environment = Environment(loader=PackageLoader('pascal_voc_tools',
                                                       'templates'),
                                  keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')
        # some peramaters
        self.size = []

    def load(self, xml_file_path):
        """Load a xml file data.

        Args:
            xml_file_path: str, the path of a xml file
        Returns:
            a dict including info in xml file.
        """
        assert os.path.exists(xml_file_path), xml_file_path

        self.file_path = xml_file_path
        tree = ET.parse(self.file_path)
        root = tree.getroot()
        self.template_parameters = self.parse_element(root)
        if 'object' not in self.template_parameters:
            self.template_parameters['object'] = []

        return self.template_parameters

    def replace_name(self, old_name, new_name):
        """Replace an object name.

        Args:
            old_name: str, an object class name.
            new_name: str, a new object class name.
        Raises:
            KeyError: if cannot find 'object' in self.template_parameters.
        """
        if 'object' not in self.template_parameters:
            raise KeyError('Make shure you have loaded an xml file data.')

        for obj in self.template_parameters['object']:
            if obj['name'] == old_name:
                obj['name'] = new_name

    def _save_element_info(self, save_dict, tag, text):
        """Save tag and text to save_dict.
        if tag not in save_dict, it will return like save_dict[tag] = text,
        otherwise it will like save_dict[tag] = [text, ]

        Args:
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

        Args:
            element: element, an element type node in xml tree.
            save_dict: dict, save result.
        Returns:
            save_dict: dict,
                like {'path': './', 'segmented': '0', 'object': [{}]}.
        """
        if save_dict is None:
            save_dict = {}

        for child in element:
            if not child:
                self._save_element_info(save_dict, child.tag, child.text)
            else:
                self._save_element_info(save_dict, child.tag,
                                        self.parse_element(child))

        return save_dict

    def set_head(self, path, width, height,
                      depth=3,
                      database='Unknown',
                      segmented=0):
        """Generate a xml file

        Args:
            path: str, arg in xml about image.
            width: int or str, image width.
            height: int or str, image height.
            depth: int or str, image channle.
            database: str, default='Unknown'.
            segmented: int or str, default=0.
        """
        abspath = os.path.abspath(path)

        self.template_parameters['path'] = abspath
        self.template_parameters['filename'] = os.path.basename(abspath)
        self.template_parameters['folder'] = os.path.basename(os.path.dirname(abspath))
        self.template_parameters['size'] = {
                'width': width,
                'height': height,
                'depth': depth,
            }
        self.template_parameters['source'] = {
                'database': database,}
        self.template_parameters['segmented'] = segmented
        self.template_parameters['object'] = []

    def add_object(self,
                   name,
                   xmin,
                   ymin,
                   xmax,
                   ymax,
                   pose='Unspecified',
                   truncated=0,
                   difficult=0):
        """add an object info

        Args:
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

        Args:
            annotation_path: str, the path of xml to save.
            image_parameters: dict, some info need to set. Default is None.s
        """
        if image_parameters is not None:
            for k, v in image_parameters.items():
                self.template_parameters[k] = v

        with open(annotation_path, 'w') as xml_file:
            content = self.annotation_template.render(
                **self.template_parameters)
            xml_file.write(content)

