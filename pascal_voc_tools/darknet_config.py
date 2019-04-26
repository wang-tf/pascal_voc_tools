#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from collections import OrderedDict


class Layer():
    """Darknet Layer Struct"""
    def __init__(self, name):
        self.name = name
        self.param = OrderedDict()


class DarknetConfig():
    def __init__(self):
        self.layers = None

    def parse(self, config_path):
        """解析darknet的config文件"""
        with open(config_path) as f:
            config_data = f.read().strip().split('\n')
            # config_data = [line.strip() for line in config_data if line]
            # config_data = [line for line in config_data if line[0] != '#']
        layers = []
        layers_str = ('\n'.join(config_data)).split('[')
        for layer_str in layers_str:
            if not layer_str:
                continue
            name, param_str = layer_str.split(']')
            param_list = param_str.strip().split('\n')
            new_layer = Layer(name)
            for param in param_list:
                if param == '' or param[0] == '#':
                    key = param
                    value = ''
                else:
                    key, value = param.strip().split('=')
                    key = key.strip()
                    if key in new_layer.param:
                        print('Warrning: The key {} has been appended, the value is {}'.format(
                            key, new_layer.param[key]))
                new_layer.param[key] = value.strip()
            layers.append(new_layer)
        self.layers = layers
        return layers

    def change_anchors(self, layers, new_anchors):
        for i, layer in enumerate(layers):
            if layer.name == 'yolo':
                mask = layer.param['mask'].split(',')
                mask = [int(s.strip()) for s in mask]
                anchors_str = []
                for mask_index in mask:
                    assert mask_index < len(new_anchors)
                anchors_str = ['{:.2f},{:.2f}'.format(
                    anchor[0], anchor[1]) for anchor in new_anchors]
                layers[i].param['anchors'] = ', '.join(anchors_str)
                layers[i].param['num'] = str(len(new_anchors))
        return layers

    def save(self, layers, save_path):
        """保存parser解析的layer结构"""
        config_data = []
        for layer in layers:
            param_list = []
            for key, val in layer.param.items():
                if key == '' or key[0] == '#':
                    param_list.append(key+' '+val)
                else:
                    param_list.append(key+' = ' + val)
            name = '['+layer.name+']'
            config_data.append(name + '\n' + '\n'.join(param_list) + '\n')
        config_data = '\n'.join(config_data)

        with open(save_path, 'w') as f:
            f.write(config_data)
        print('Data have been saved in {}.'.format(save_path))
        return 0

    def check_yolo(self, cfg_file, label_list_file):
        assert os.path.exists(cfg_file), cfg_file
        assert os.path.exists(label_list_file), label_list_file

        # get labels
        with open(label_list_file) as f:
            labels = f.read().strip().split('\n')

        layers = self.parse(cfg_file)
        for i in range(len(layers)-1, 1):
            if layers[i].name == 'yolo':
                # check classes
                assert int(layers[i].param['classes']) == len(labels)
                # check anchors number
                anchors = layers[i].param['anchors'].split(',')
                assert anchors % 2 == 0, 'anchors length is not right: {}'.format(
                    anchors)
                # check num
                assert int(layers[i].param['num']) == len(anchors/2)
                # check mask
                mask = list(map(int, layers[i].param['mask'].split(',')))
                for m in mask:
                    assert m < anchors/2, 'mask is not right: {}'.format(mask)
                # check filters
                assert layers[i-1].name == 'convolutional'
                assert int(layers[i-1].param['filters']) == (len(labels) + 5) * len(
                    mask), 'filters is not right: {}'.format(layers[i-1].param['filters'])
        print('Check Done.')
        return 0


def main():
    from kmeans import KMeansAnchors
    cfg_file = './configs/yolov3-20190423.cfg'
    config = DarknetConfig()
    layers = config.parse('./configs/yolov3-20190423.cfg-backup')
    width = layers[0].param['width']
    yolo_input_shape = int(width)
    kmeans = KMeansAnchors()
    anchors = kmeans.calculate(yolo_input_shape=yolo_input_shape)
    print("Result: {}".format(anchors))
    layers = config.change_anchors(layers, anchors)
    config.save(layers, cfg_file)


if __name__ == '__main__':
    main()
