# -*- coding:utf-8 -*-
"""
decode xml file of Pascal Voc annotation write a xml file about pascal voc annotation.
"""
import os
import logging
from jinja2 import Environment, PackageLoader
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from lxml import etree
from lxml.etree import SubElement

logger = logging.getLogger(__name__)

class Size(object):
    """Size data format in Pascal VOC xml
    """
    def __init__(self, width: int = 0, height: int = 0, depth: int = 0):
        self.width = width
        self.height = height
        self.depth = depth


class Source(object):
    def __init__(self, database: str = ''):
        self.database = database


class Bndbox(object):
    """Bndbox data format in Object for Pascal VOC xml
    """
    def __init__(self,
                 xmin: int = 0,
                 ymin: int = 0,
                 xmax: int = 0,
                 ymax: int = 0):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


class XmlObject(object):
    """Object data foramt in Pascal VOC xml
    """
    def __init__(self,
                 name: str = '',
                 pose: str = 'Unspecified',
                 truncated: int = 0,
                 difficult: int = 0,
                 bndbox: Bndbox = Bndbox()):
        self.name = name
        self.pose = pose
        self.truncated = truncated
        self.difficult = difficult
        self.bndbox = bndbox


class PascalXml(object):
    """Pascal VOC xml file data format
    """
    def __init__(self,
                 folder: str = '',
                 filename: str = '',
                 path: str = '',
                 source: Source = Source(),
                 size: Size = Size(),
                 segmented: int = 0,
                 object: list = []):
        self.folder = folder
        self.filename = filename
        self.path = path
        self.source = source
        self.size = size
        self.segmented = segmented
        self.object = object

    def load(self, xml_file_path):
        """form a xml file load data.
        """
        return load_pascal_xml(xml_file_path, self)

    def save(self, save_xml_path):
        """save data to a xml file.
        """
        return save_pascal_xml(save_xml_path, self)

    def replace_name(self, old_name, new_name):
        """Replace an object name.

        Args:
            old_name: str, an object class name.
            new_name: str, a new object class name.
        """
        for obj in self.object:
            if obj.name == old_name:
                obj.name = new_name
        return self


def load_pascal_xml(
    xml_file_path: str, default_format=PascalXml()) -> PascalXml:
    """Load a pascal format xml file.
    Arguments:
        xml_file_path: a xml file path.
        default_format: PascalXml instance.
    Returns:
        default_format: the PascalXml instance including data.
    Raises:
        ListError: can not find the key in xml file.
    """
    html = etree.parse(xml_file_path)
    annotation = html.xpath('/annotation')[0]  # load first annotation

    default_format.folder = annotation.xpath('//folder/text()')[0]
    default_format.filename = annotation.xpath('//filename/text()')[0]
    try:
        default_format.path = annotation.xpath('//path/text()')[0]
    except Exception:
        logger.warning('Can not find path node in xml.')
        pass
    try:
        default_format.source.database = annotation.xpath('//source/database/text()')[0]
    except Exception:
        logger.warning('Can not find source/database node in xml.')
        pass

    default_format.size.width = int(annotation.xpath('//size/width/text()')[0])
    default_format.size.height = int(
        annotation.xpath('//size/height/text()')[0])
    default_format.size.depth = int(annotation.xpath('//size/depth/text()')[0])
    default_format.segmented = int(annotation.xpath('//segmented/text()')[0])

    for obj in annotation.xpath('//object'):
        xml_obj = XmlObject()
        xml_obj.name = obj.xpath('//name/text()')[0]
        try:
            xml_obj.pose = obj.xpath('//pose/text()')[0]
        except Exception:
            logger.warning("Can not find pose node in xml")
            pass
        xml_obj.truncated = int(obj.xpath('//truncated/text()')[0])
        xml_obj.difficult = int(obj.xpath('//difficult/text()')[0])
        xml_obj.bndbox.xmin = int(obj.xpath('//bndbox/xmin/text()')[0])
        xml_obj.bndbox.ymin = int(obj.xpath('//bndbox/ymin/text()')[0])
        xml_obj.bndbox.xmax = int(obj.xpath('//bndbox/xmax/text()')[0])
        xml_obj.bndbox.ymax = int(obj.xpath('//bndbox/ymax/text()')[0])
        default_format.object.append(xml_obj)

    return default_format


def save_pascal_xml(save_xml_path: str, pascal_xml: PascalXml) -> PascalXml:
    """save data to xml
    Arguments:
        save_xml_path: save path.
        pascal_xml: PascalXml.
    Returns:
        PascalXml
    """
    node_root = etree.Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = pascal_xml.folder

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = pascal_xml.filename

    node_path = SubElement(node_root, 'path')
    node_path.text = pascal_xml.path

    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = pascal_xml.source.database

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(pascal_xml.size.width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(pascal_xml.size.height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(pascal_xml.size.depth)

    for obj in pascal_xml.object:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = obj.name
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = str(obj.difficult)
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(obj.bndbox.xmin)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(obj.bndbox.ymin)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(obj.bndbox.xmax)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(obj.bndbox.ymax)

    tree = etree.ElementTree(node_root)
    tree.write(save_xml_path,
               pretty_print=True,
               xml_declaration=False,
               encoding='utf-8')

    return pascal_xml


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

    def set_head(self,
                 path,
                 width,
                 height,
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
        self.template_parameters['folder'] = os.path.basename(
            os.path.dirname(abspath))
        self.template_parameters['size'] = {
            'width': width,
            'height': height,
            'depth': depth,
        }
        self.template_parameters['source'] = {
            'database': database,
        }
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
