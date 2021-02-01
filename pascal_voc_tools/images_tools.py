# coding:utf-8

import os
import glob


class JPEGImages(object):
    def __init__(self, img_dir):
        self.dir = img_dir
        self.jpg_list = []
    
    def load(self):
        self.jpg_list = sorted(glob.glob(os.path.join(self.jpg_dir, '*.jpg')))
        return self
    