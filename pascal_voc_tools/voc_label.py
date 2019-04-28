#!/usr/bin/env python3

"""
@Author: wangtf
@Description:
Generating useful files for darknet using pascal voc dataset
"""
import xml.etree.ElementTree as ET
import pickle
import os


class DarknetDataset():
    def __init__(self):
        self.voc_sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"]

    def get_classes(self, voc_root_path):
        # The labels.list file mast be in voc_root_path dir.
        label_path = os.path.join(voc_root_path, 'labels.list')
        assert os.path.exists(label_path), label_path

        with open(label_path) as f:
            labels = f.read().strip().split('\n')
            labels = [label.strip() for label in labels]
        self.classes = labels
        print('Classes change to: {}'.format(self.classes))
        return self.classes

    def convert(self, size, box):
        dw = 1./(size[0])
        dh = 1./(size[1])
        x = (box[0] + box[1])/2.0 - 1
        y = (box[2] + box[3])/2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x, y, w, h)

    def convert_annotation(self, year, image_id, voc_root_path):
        in_file = open(os.path.join(
            voc_root_path, 'VOC%s/Annotations/%s.xml' % (year, image_id)))
        out_file = open(os.path.join(
            voc_root_path, 'VOC%s/labels/%s.txt' % (year, image_id)), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes or int(difficult) == 1:
                continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
                xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = self.convert((w, h), b)
            out_file.write(str(cls_id) + " " +
                           " ".join([str(a) for a in bb]) + '\n')

    def voc2(self, voc_root_path):
        # change labels
        self.get_classes(voc_root_path)

        for year, image_set in self.voc_sets:
            if not os.path.exists(os.path.join(voc_root_path, 'VOC%s/labels/' % (year))):
                os.makedirs(os.path.join(
                    voc_root_path, 'VOC%s/labels/' % (year)))
            if not os.path.exists(os.path.join(voc_root_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set))):
                continue

            image_ids = open(os.path.join(
                voc_root_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set))).read().strip().split()
            with open(os.path.join(voc_root_path, '%s_%s.txt' % (year, image_set)), 'w') as list_file:
                image_file_list = []
                for image_id in image_ids:
                    image_file_path = os.path.abspath(os.path.join(
                        voc_root_path, 'VOC{}/JPEGImages/{}.jpg'.format(year, image_id)))
                    image_file_list.append(image_file_path)
                    self.convert_annotation(year, image_id, voc_root_path)
                list_file.write('\n'.join(image_file_list))

        #os.system("cat {0}/2007_train.txt {0}/2007_val.txt > {0}/train.txt".format(voc_root_path))
        #os.system("cat {0}/2007_train.txt {0}/2007_val.txt {0}/2007_test.txt > {0}/train.all.txt".format(voc_root_path))
