#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os


class CropData():
    def __init__(self, root_dir, save_root_dir):
        self.root_dir = root_dir
        self.save_root_dir = save_root_dir
        

    def crop(image_path, xml_path, )