# -*- coding:utf-8 -*-

import os
import json
from pascal_voc_tools._dataset_tools import COCODataset
from pascal_voc_tools._dataset_tools import DarknetDataset


class DetectionResultSaver():
    def __init__(self, save_format='coco', label_file_path=''):
        self.format = save_format

        if self.format == 'coco':
            self.dataset = self._set_coco_head()
        elif self.format == 'darknet':
            self.dataset = self._set_darknet_head(label_file_path)
        elif self.format == 'pascalvoc':
            pass
        else:
            print(
                'ERROR: the save_format {} is unknown, please choose from [coco, darknet, pascalvoc]'
                .format(save_format))
            raise

    def _set_coco_head(self):
        return COCODataset()

    def _set_darknet_head(self, label_file_path):
        return DarknetDataset(label_file_path)

    def add_image_coco(self, file_name, image_id, width, height,
                       image_license):
        self.dataset.add_image(file_name, image_id, width, height,
                               image_license)

    def add_bbox_coco(self, image_id, annotation_id, category_id, bbox):
        self.dataset.add_annotation(annotation_id, image_id, category_id, bbox)

    def add_one_obj_coco(self, image_path, category_name, bbox):
        image_id = self.dataset.get_image_id_by_image_path(image_path)
        category_id = self.dataset.get_category_id_by_name(category_name)
        annotation_id = self.dataset.annotation_next_new_id
        self.dataset.annotation_next_new_id += 1
        self.dataset.add_annotation(annotation_id, image_id, category_id, bbox)

    def save_coco(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.dataset, f)

    def add_one_obj_darknet(self, image_name_id, category_name, confidence, bbox):
        self.dataset.add_annotation(image_name_id, category_name, confidence, bbox)

    def save_darknet(self, save_dir, image_set='test'):
        for category_name in self.dataset.data.keys():
            save_file_name = 'comp4_det_{}_{}.txt'.format(image_set, category_name)

            with open(save_file_name, 'w') as f:
                f.write('\n'.join(list(map(str, self.dataset.data[category_name]))))
            print('INFO: saved {}'.format(save_file_name))