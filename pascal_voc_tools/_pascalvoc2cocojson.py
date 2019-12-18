# -*- coding:utf-8 -*-

import os
import datetime
import cv2
from pascal_voc_tools import XmlParser

TIME_NOW = datetime.datetime.now()
ANNOTATION_INIT_ID = 0

class COCODataset():
    def __init__(self):
        self.image_init_id = 0
        self.annotation_init_id = 0
        self.time_now = datetime.datetime.now()

        self.image_current_max_id = self.image_init_id
        self.annotation_current_max_id = self.annotation_init_id

        self.image_name_id_map = {}
        self.category_name_id_map = {}

    @staticmethod
    def info(self, year=TIME_NOW.year, version='v1', description='dataset',
        contributor='', url='', date_crtated=TIME_NOW):
        coco_info = {
            'year': year,
            "version": version,
            "description": description,
            "contributor": contributor,
            "url": url,
            "date_created": str(date_crtated)
        }
        return coco_info

    @staticmethod
    def license(self, license_id=1, name='', url=''):
        coco_license = {
            "id": license_id,
            "name": name,
            "url": url
        }
        return coco_license


coco_info = {
    "year": TIME_NOW.year,
    "version": "v1",
    "description": "dataset",
    "contributor": "nobody",
    "url": "",
    "date_created": str(TIME_NOW)}


coco_license = {
    "id": 1,
    "name": "",
    "url": ""
}


coco_image = {
    "coco_url": "",
    "date_captured": "",
    "file_name": "",
    "flickr_url": "",
    "id": 0,
    "height": 0,
    "width": 0,
    "license": 1
}


coco_annotation = {
    "id": 0,
    "image_id": 0,
    "category_id": 0,
    "segmentation": [],
    "area": 0,  # width * height
    "bbox": [],  # [xmin, ymin, width, height]
    "iscrowd": 0
}


coco_category = {
    "id": 0,
    "name": "",
    "supercategory": ""
}


def add_image(image_list, file_name, image_id, width, height, image_license=1):
    new_image = coco_image.copy()
    new_image['file_name'] = file_name
    new_image['id'] = image_id
    new_image['width'] = width
    new_image['height'] = height
    new_image['license'] = image_license
    
    image_list.append(new_image)


def get_category_id(category_list, category_name, supercategory=None):
    category_id = -1
    for category in category_list:
        if category['name'] == category_name:
            category_id = category['id']
    
    if category_id == -1:
        new_category = coco_category.copy()
        category_id = len(category_list)
        new_category['id'] = category_id
        new_category['name'] = category_name
        new_category['supercategory'] = category_name if supercategory is None else supercategory

        category_list.append(new_category)
    return category_id


def add_annotation(annotation_list, image_id, annotation_id, category_id, bbox):
    new_annotation = coco_annotation.copy()
    new_annotation['id'] = annotation_id
    new_annotation['image_id'] = image_id
    new_annotation['category_id'] = category_id
    new_annotation['bbox'] = bbox
    new_annotation['area'] = bbox[2] * bbox[3]

    annotation_list.append(new_annotation)


def pascalvoc2json(voc_root_dir, main_set_name='train'):
    xml_dir = os.path.join(voc_root_dir, 'Annotations')
    jpg_dir = os.path.join(voc_root_dir, 'JPEGImages')
    main_set_dir = os.path.join(voc_root_dir, 'ImageSets/Main')

    main_set_path = os.path.join(main_set_dir, main_set_name+'.txt')

    with open(main_set_path) as f:
        lines = f.read().strip().split('\n')
    image_ids = [line.strip() for line in lines]

    json_content = {
        "info": coco_info,
        "licenses": [coco_license],
        "images": [],
        "annotations": [],
        "categories": []
    }

    annotation_id = ANNOTATION_INIT_ID
    for image_id, name_id in enumerate(image_ids):
        # load image
        image_path = os.path.join(jpg_dir, name_id+'.jpg')
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        add_image(json_content['images'], name_id+'.jpg', image_id, width, height)

        # load annotation
        xml_file_path = os.path.join(xml_dir, name_id+'.xml')
        xml_data = XmlParser().load(xml_file_path)

        for obj in xml_data['object']:
            category_name = obj['name']
            # 
            category_id = get_category_id(json_content['categories'], category_name)

            xmin = float(obj['bndbox']['xmin'])
            ymin = float(obj['bndbox']['ymin'])
            xmax = float(obj['bndbox']['xmax'])
            ymax = float(obj['bndbox']['ymax'])
            add_annotation(json_content['annotations'], image_id, annotation_id, category_id, [xmin, ymin, xmax-xmin, ymax-ymin])
            annotation_id += 1  # reset annotation id

    return json_content
