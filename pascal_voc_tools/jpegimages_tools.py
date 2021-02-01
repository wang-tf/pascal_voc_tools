# coding:utf-8

import os
import glob
import logging

logger = logging.getLogger(__name__)


class JPEGImages(object):
    """Some tools for JPEGImages dir.
    """
    def __init__(self, image_dir=None):
        self.dir = image_dir
        self.jpg_list = []

    def load(self, image_dir=None):
        """Load jpg image files' path.

        Arguments:
            image_dir: a directory incluing jpg images.
        """
        if image_dir:
            self.dir = image_dir

        assert os.path.isdir(self.dir), self.dir
        self.jpg_list = sorted(glob.glob(os.path.join(self.dir, '*.jpg')))

        logger.info(f'Load {len(self.jpg_list)} images.')
        return self
