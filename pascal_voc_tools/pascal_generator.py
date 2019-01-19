#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
from six import raise_from
from PIL import Image

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import pascal_voc_tools  # noqa: F401
    __package__ = "pascal_voc_tools"

from .xmlreader import XmlReader
from .generator import Generator
from ..utils.image import read_image_bgr


class PascalVocGenerator(Generator):
    """ Generate data for a Pascal VOC dataset.

    See http://host.robots.ox.ac.uk/pascal/VOC/ for more information.
    """

    def __init__(
        self,
        data_dir,
        set_name,
        classes,
        image_extension='.jpg',
        skip_truncated=False,
        skip_difficult=False,
        **kwargs
    ):
        """ Initialize a Pascal VOC data generator.

        Args
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
            csv_class_file: Path to the CSV classes file.
        """
        self.data_dir             = data_dir
        self.set_name             = set_name
        self.classes              = classes
        self.image_names          = [l.strip().split(None, 1)[0] for l in open(os.path.join(data_dir, 'ImageSets', 'Main', set_name + '.txt')).readlines()]
        self.image_extension      = image_extension
        self.skip_truncated       = skip_truncated
        self.skip_difficult       = skip_difficult

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(PascalVocGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        path  = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        image = Image.open(path)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        return read_image_bgr(path)

    def __format_annotation(self, annotation):
        """ Parse an annotation given an XML element.
        """
        truncated = int(annotation['truncated'])
        difficult = int(annotation['difficult'])

        class_name = annotation['name']
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        box = np.zeros((4,))
        label = self.name_to_label(class_name)

        bndbox    = annotation['bndbox']
        box[0] = float(bndbox['xmin']) - 1
        box[1] = float(bndbox['ymin']) - 1
        box[2] = float(bndbox['xmax']) - 1
        box[3] = float(bndbox['ymax']) - 1

        return truncated, difficult, box, label

    def __format_annotations(self, annotations):
        """ Parse all annotations under the xml_root.
        """
        annotations = {'labels': np.empty((len(annotations['object']),)), 'bboxes': np.empty((len(annotations['object']), 4))}
        for i, element in enumerate(annotations['object']):
            try:
                truncated, difficult, box, label = self.__format_annotation(element)
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

            if truncated and self.skip_truncated:
                continue
            if difficult and self.skip_difficult:
                continue

            annotations['bboxes'][i, :] = box
            annotations['labels'][i] = label

        return annotations

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        filename = self.image_names[image_index] + '.xml'
        try:
            xml_reader = XmlReader(os.path.join(self.data_dir, 'Annotations', filename))
            annotations = xml_reader.parse_annotations()
            return self.__format_annotations(annotations)
