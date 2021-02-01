# -*- coding:utf-8 -*-
"""
decode xml file of Pascal Voc annotation write a xml file
about pascal voc annotation.
"""
import logging

from lxml import etree
from lxml.etree import SubElement

from .tools import bb_intersection_over_union as iou

logger = logging.getLogger(__name__)


class ImageSize(object):
    """Size data format in Pascal VOC xml

    Attributes:
        width: a int of image width.
        height: a int of image height.
        depth: a int of image depth.
    """
    def __init__(self, width: int = 0, height: int = 0, depth: int = 0):
        self.width = width
        self.height = height
        self.depth = depth

    def __str__(self):
        return f"ImageSize({self.width}, {self.height}, {self.depth})"


class DataSource(object):
    """Source data format in Pascal VOC xml

    Attributes:
        database: a str of dataset name.
    """
    def __init__(self, database: str = ''):
        self.database = database

    def __str__(self):
        return f"DataScorce({self.database})"


class Bndbox(object):
    """Bndbox data format in Object for Pascal VOC xml

    Attributes:
        xmin: a int of bndbox left.
        ymin: a int of bndbox top.
        xmax: a int of bndbox right.
        ymax: a int of bndbox bottom.
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

    def __str__(self):
        return f"Bndbox({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax})"

    def convert2relative_xywh(self, size):
        """from absolute coordinate to relative coordinate

        Arguments:
            size: a tuple of width and height
        """
        box = (float(self.xmin), float(self.xmax), float(self.ymin),
               float(self.ymax))
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = max((box[0] + box[1]) / 2.0 - 1, 0)
        y = max((box[2] + box[3]) / 2.0 - 1, 0)
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def resize(self, rate, horizion_bias=0, vertical_bias=0):
        """resize a bbox

        Args:
            rate: box change rate;
            horizion_bias: xmin and xmax will add it;
            vertical_bias: ymin and ymax will add it,
        Returns:
            new bbox list
        """
        self.xmin = self.xmin * rate + horizion_bias
        self.ymin = self.ymin * rate + vertical_bias
        self.xmax = self.xmax * rate + horizion_bias
        self.ymax = self.ymax * rate + vertical_bias
        return self


class XmlObject(object):
    """Object data foramt in Pascal VOC xml

    Attributes:
        name: the category name.
        pose: object post description.
        truncated: default is 0.
        difficult: default is 0.
        bndbox: the Bndbox.
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

    def __str__(self):
        return f"XmlObject({self.name}, {self.pose}," + \
            f" {self.truncated}, {self.difficult}, {self.bndbox})"


class PascalXml(object):
    """Pascal VOC xml file data format

    Attributes:
        folder: image folder.
        filename: image file name.
        path: image file path.
        source: the DataSource.
        size: the ImageSize.
        segmented: default 0.
        object_list: a list of XmlObject
    """
    def __init__(self,
                 folder: str = '',
                 filename: str = '',
                 path: str = '',
                 source: DataSource = DataSource(),
                 size: ImageSize = ImageSize(),
                 segmented: int = 0,
                 object_list: list = []):
        self.folder = folder
        self.filename = filename
        self.path = path
        self.source = source
        self.size = size
        self.segmented = segmented
        self.object = object_list

    def load(self, xml_file_path):
        """form a xml file load data.

        Arguments:
            xml_file_path: a xml file path.
        """
        load_pascal_xml(xml_file_path, self)
        return self

    def save(self, save_xml_path):
        """save data to a xml file.

        Arguments:
           save_xml_path: the xml path to save data.
        """
        save_pascal_xml(save_xml_path, self)
        return self

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

    def convert2yolotxt(self, save_path: str, classes: list):
        """save data to txt for yolo format.

        Arguments:
            save_path: the txt file path to save data.
            classes: a list of categories.
        """
        assert save_path[
            -4:] == '.txt', f"Please check save_path is right: {save_path}"

        out_file = open(save_path, 'w')

        w = int(self.size.width)
        h = int(self.size.height)

        for obj in self.object:
            difficult = obj.difficult
            cls = obj.name
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            bb = obj.bndbox.convert2relative_xywh((w, h))
            out_file.write(
                str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

        out_file.close()

        return self

    def convert2csv(self, classes: list = []):
        """Convert data to info list. The head is
        [file_name, width, height, category, xmin, ymin, xmax, ymax]

        Arguments:
            classes: a list of categories.
        """
        file_name = self.filename
        width = self.size.width
        height = self.size.height

        info_list = []
        for bbox in self.object:
            category = bbox.name
            xmin = bbox.bndbox.xmin
            ymin = bbox.bndbox.ymin
            xmax = bbox.bndbox.xmax
            ymax = bbox.bndbox.ymax

            if classes and category not in classes:
                continue
            info_list.append(
                [file_name, width, height, category, xmin, ymin, xmax, ymax])
        return info_list

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
        bndbox = Bndbox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        obj = XmlObject(name=name,
                        pose=pose,
                        truncated=truncated,
                        difficult=difficult,
                        bndbox=bndbox)
        self.object.append(obj)
        return self

    def resize_obj_by_rate(self, rate: float):
        """Resize all bndbox by rate.

        Arguments:
            rate: a float for timeing bndbox.
        """
        original_width = int(self.size.width)
        original_height = int(self.size.height)
        new_width = int(original_width * rate)
        new_height = int(original_height * rate)

        horizion_bias = 0
        vertical_bias = 0

        self.size.width = new_width
        self.size.height = new_height

        for obj in self.object:
            obj.bndbox = obj.bndbox.resize(rate, horizion_bias, vertical_bias)

        return self

    def crop_annotations(self, split_bboxes, iou_thresh=0.7):
        """Using split_bboxes to split an xml file.
        Arguments:
            xml_info: dict, all info about a xml.
            split_bboxes: list, like [[xmin, ymin, xmax, ymax], ]
        Returns:
            subannotations: list, like [xml_info, ]
        """
        subannotations = []
        for bbox in split_bboxes:
            xmin, ymin, xmax, ymax = bbox

            # init sub xml info
            sub_xml = PascalXml()
            sub_xml.folder = self.folder
            sub_xml.path = self.path
            sub_xml.filename = self.filename
            sub_xml.size = ImageSize(xmax - xmin, ymax - ymin, self.size.depth)
            sub_xml.source = DataSource(self.source.database)
            sub_xml.segmented = self.segmented
            sub_xml.object = []
            for bbox in self.object:
                ob_xmin = bbox.bndbox.xmin
                ob_ymin = bbox.bndbox.ymin
                ob_xmax = bbox.bndbox.xmax
                ob_ymax = bbox.bndbox.ymax

                if iou([ob_xmin, ob_ymin, ob_xmax, ob_ymax],
                       [xmin, ymin, xmax, ymax]) > iou_thresh:
                    sub_bbox = Bndbox(max(ob_xmin - xmin, 1),
                                      max(ob_ymin - ymin, 1),
                                      min(ob_xmax - xmax, xmax - xmin - 1),
                                      min(ob_ymax - ymax, ymax - ymin - 1))
                    sub_obj = XmlObject(name=bbox.name,
                                        bndbox=sub_bbox,
                                        truncated=bbox.truncated,
                                        difficult=bbox.difficult)
                    sub_xml.object.append(sub_obj)
            subannotations.append(sub_xml)

        return subannotations


def load_pascal_xml(
                    xml_file_path: str,
                    default_format=PascalXml()) -> PascalXml:
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

    default_format.folder = annotation.xpath('./folder/text()')[0]
    default_format.filename = annotation.xpath('./filename/text()')[0]
    try:
        default_format.path = annotation.xpath('./path/text()')[0]
    except Exception:
        logger.warning('Can not find path node in xml.')
        pass
    try:
        default_format.source = DataSource()
        default_format.source.database = annotation.xpath(
            './source/database/text()')[0]
    except Exception:
        logger.warning('Can not find source/database node in xml.')
        pass

    default_format.size = ImageSize()
    default_format.size.width = int(annotation.xpath('./size/width/text()')[0])
    default_format.size.height = int(
        annotation.xpath('./size/height/text()')[0])
    default_format.size.depth = int(annotation.xpath('./size/depth/text()')[0])
    default_format.segmented = int(annotation.xpath('./segmented/text()')[0])

    default_format.object = []
    for obj in annotation.xpath('./object'):
        xml_obj = XmlObject()
        xml_obj.name = obj.xpath('./name/text()')[0]
        try:
            xml_obj.pose = obj.xpath('./pose/text()')[0]
        except Exception:
            logger.warning("Can not find pose node in xml")
            pass
        xml_obj.truncated = int(obj.xpath('./truncated/text()')[0])
        xml_obj.difficult = int(obj.xpath('./difficult/text()')[0])
        xml_obj.bndbox = Bndbox()
        xml_obj.bndbox.xmin = int(obj.xpath('./bndbox/xmin/text()')[0])
        xml_obj.bndbox.ymin = int(obj.xpath('./bndbox/ymin/text()')[0])
        xml_obj.bndbox.xmax = int(obj.xpath('./bndbox/xmax/text()')[0])
        xml_obj.bndbox.ymax = int(obj.xpath('./bndbox/ymax/text()')[0])
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
        raise ("Please use new class PascalXml")
