# -*- coding:utf-8 -*-

from pascal_voc_tools._pascalvoc2cocojson import COCODataset
from pascal_voc_tools._pascalvoc2cocojson import add_annotation
from pascal_voc_tools._pascalvoc2cocojson import add_image


class DetectionResultSaver():
    def __init__(self, save_format='coco'):
        self.format = save_format

        if self.format == 'coco':
            self.data = self.set_coco_head()

    def set_coco_head(self):
        self.data = {
            "info": COCODataset.info(),
            "licenses": [COCODataset.license()],
            "images": [],
            "annotations": [],
            "categories": []
        }

    def add_image_coco(self, file_name, image_id, width, height, image_license):
        add_image(self.data['images'], file_name, image_id, width, height, image_license)
    
    def add_bbox_coco(self, image_id, annotation_id, category_id, bbox):
        add_annotation(self.data['annotations'], image_id, annotation_id, category_id, bbox)

    def add_one_obj_coco(self, image_name, category_name, bbox):
        self.image_name_id_map = {}
        