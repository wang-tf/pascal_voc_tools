#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import os
import random

__DEBUG__ = False


def width_and_height_iou(x, centroids):
    """
    Arguments:
        x: 某一个ground truth的w,h;
        centroids: anchor的w,h的集合[(w,h),(),...]，共k个;
    Return:
        单个ground truth box与所有k个anchor box的IoU值集合;
    """
    IoUs = []
    w, h = x  # ground truth的w,h
    for centroid in centroids:
        c_w, c_h = centroid  # anchor的w,h
        if c_w >= w and c_h >= h:  # anchor包围ground truth
            iou = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:  # anchor宽矮
            iou = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:  # anchor瘦长
            iou = c_w*h/(w*h + c_w*(c_h-h))
        else:  # ground truth包围anchor     means both w,h are bigger than c_w and c_h respectively
            iou = (c_w*c_h)/(w*h)
        IoUs.append(iou)  # will become (k,) shape
    return np.array(IoUs)


def avg_IOU(X, centroids):
    '''
    Arguments:
        X: ground truth的w,h的集合[(w,h),(),...]
        centroids: anchor的w,h的集合[(w,h),(),...]，共k个
    Return:
        centroids与ground truth 的平均iou
    '''
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # 返回一个ground truth与所有anchor的IoU中的最大值
        sum += max(width_and_height_iou(X[i], centroids))
    return sum/n  # 对所有ground truth求平均


def write_anchors_to_file(anchors, X, anchor_file):
    '''
    Arguments:
        centroids: anchor的w,h的集合[(w,h),(),...]，共k个
        X: ground truth的w,h的集合[(w,h),(),...]
        anchor_file: anchor和平均IoU的输出路径
    Return:
        anchors: saved anchors.
    '''
    with open(anchor_file, 'w') as f:
        anchors_str = []
        for i in len(anchors):
            anchors_str.append('{:.2f},{:.2f}'.format(
                anchors[i, 0], anchors[i, 1]))
        f.write(', '.join(anchors_str) + '\n')
        f.write('%f\n' % (avg_IOU(X, anchors)))


def kmeans(ground_truth_list, centroids):
    """以长宽的iou作为距离度量，以centroids为质心聚类
    Arguments:
        ground_truth_list: 所有ground truth的长宽
        centroids: 初始化的质心
    """
    ground_truth_number = ground_truth_list.shape[0]
    print("centroids.shape", centroids)
    anchor_number, anchor_dim = centroids.shape  # anchor的个数k以及w,h两维，dim默认等于2
    prev_assignments = np.ones(ground_truth_number) * \
        (-1)  # 对每个ground truth分配初始标签
    iterator = 0
    # 初始化每个ground truth对每个anchor的IoU
    old_all_distance = np.zeros((ground_truth_number, anchor_number))

    while True:
        all_distance = []
        iterator += 1
        for i in range(ground_truth_number):
            distance = 1 - \
                width_and_height_iou(ground_truth_list[i], centroids)
            all_distance.append(distance)
        # 得到每个ground truth对每个anchor的IoU
        all_distance = np.array(all_distance)

        # 计算每次迭代和前一次IoU的变化值
        print("iter {}: dists = {}".format(iterator, np.sum(
            np.abs(old_all_distance-all_distance))), end='\r')

        # assign samples to centroids
        # 将每个ground truth分配给距离d最小的anchor序号
        assignments = np.argmin(all_distance, axis=1)

        # 如果前一次分配的结果和这次的结果相同，就输出anchor以及平均IoU
        if (assignments == prev_assignments).all():
            break

        # calculate new centroids
        # 初始化以便对每个簇的w,h求和
        centroid_sums = np.zeros((anchor_number, anchor_dim), np.float)
        # 将每个簇中的ground truth的w和h分别累加
        for i in range(ground_truth_number):
            centroid_sums[assignments[i]] += ground_truth_list[i]
        # 对簇中的w,h求平均
        for j in range(anchor_number):
            centroids[j] = centroid_sums[j]/(np.sum(assignments == j)+1)

        prev_assignments = assignments.copy()
        old_all_distance = all_distance.copy()

    return centroids


class AnchorsKMeans():
    """对anchors聚类
    """

    def __init__(self, filelist):
        """
        Arguments:
            filelist: path to filelist;
        """
        self.filelist = filelist
        self.anchors = None
        self.ground_truth_list = None

    def save(self, output_dir):
        # make sure the output_dir is exist.
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        anchor_file = os.path.join(output_dir, 'anchors%d.txt' %
                                   (len(self.anchors)))  # save path
        write_anchors_to_file(
            self.anchors, self.ground_truth_list, anchor_file)

    def calculate(self, num_clusters=9, yolo_version='yolov3', yolo_input_shape=416):
        """
        Arguments:
            num_clusters: number of clusters;
            yolo_version: default='yolov3', yolov2 or yolov3;
            yolo_input_shape: default=1056, input images shape，multiples of 32. etc. 416*416;

        """
        # read filelist
        with open(self.filelist) as f:
            lines = [line.rstrip('\n') for line in f.readlines()]

        # get w and h for filelist
        annotation_dims = []
        for i, line in enumerate(lines):
            line = line.replace('JPEGImages', 'labels')
            line = line.replace('.jpg', '.txt')
            line = line.replace('.png', '.txt')
            print(str(i)+': '+line, end='\r')
            with open(line) as f2:
                for line in f2.readlines():
                    line = line.rstrip('\n')
                    w, h = line.split(' ')[3:]
                    annotation_dims.append((float(w), float(h)))
        print('\n')
        annotation_dims = np.array(annotation_dims)  # 保存所有ground truth框的(w,h)

        indices = [random.randrange(annotation_dims.shape[0])
                   for i in range(num_clusters)]
        centroids = annotation_dims[indices]
        centroids = kmeans(annotation_dims, centroids)

        anchors = centroids.copy()
        if yolo_version == 'yolov2':
            for i in range(anchors.shape[0]):
                # yolo中对图片的缩放倍数为32倍，所以这里除以32，
                # 如果网络架构有改变，根据实际的缩放倍数来
                # 求出anchor相对于缩放32倍以后的特征图的实际大小（yolov2）
                anchors[i][0] *= yolo_input_shape/32.
                anchors[i][1] *= yolo_input_shape/32.
        elif yolo_version == 'yolov3':
            for i in range(anchors.shape[0]):
                # 求出yolov3相对于原图的实际大小
                anchors[i][0] *= yolo_input_shape
                anchors[i][1] *= yolo_input_shape
        else:
            print("the yolo version is not right!")
            exit(-1)

        widths = anchors[:, 0]
        sorted_indices = np.argsort(widths)

        print('Anchors = ', anchors[sorted_indices])
        self.anchors = anchors[sorted_indices]
        return self.anchors


def main():
    kmeans = AnchorsKMeans('./data/VOCdevkit/2007_train.txt')
    kmeans.calculate()


if __name__ == "__main__":
    main()
