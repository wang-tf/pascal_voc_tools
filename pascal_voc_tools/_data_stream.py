# -*- coding:utf-8 -*-

import cv2


class VideoStream():
    def __init__(self, video_path):
        self.cap = self.load_video(video_path)
        self.fps = 0
        self.size = (0, 0)
        self.video_writer = None

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        if not cap.isOpened():
            print('ERROR: can not open video_path: {}'.format(video_path))
            raise
        return cap

    def set_save_video(self, save_path):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(save_path, fourcc, self.fps, self.size)

    def get_frame(self):
        while self.cap.isOpened():
            _, image = cap.read()
            yield image

    def save_frame(self, image):
        self.video_writer.write(image)


class ImageListStream():
    pass
