# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
# sys.path.append("../")
import tensorflow as tf
import time
import cv2
import numpy as np
from help_utils.tools import *
from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from custom_libs.networks import build_whole_network
from libs.box_utils import draw_box_in_img_inference


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
class R2CNN:
    def __init__(self, model_path, gpu=0):
        self.model_path = model_path
        self.init_op = None
        self.restorer = None
        self.restore_ckpt = None
        self.config = None
        self.sess = None
        self.gpu = str(gpu)
        self.det_net = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=False)

        self.img_plac = None
        self.img_batch = None
        self.det_boxes_h = None
        self.det_scores_h = None
        self.det_category_h = None
        self.det_boxes_r = None
        self.det_scores_r = None
        self.det_category_r = None

        self.init_model()

    def init_model(self):


        # 1. preprocess img
        self.img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not GBR
        self.img_batch = tf.cast(self.img_plac, tf.float32)
        self.img_batch = self.img_batch - tf.constant(cfgs.PIXEL_MEAN)
        self.img_batch = short_side_resize_for_inference_data(img_tensor=self.img_batch,
                                                         target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                         is_resize=False)

        self.det_boxes_h, self.det_scores_h, self.det_category_h, \
        self.det_boxes_r, self.det_scores_r, self.det_category_r = self.det_net.build_whole_detection_network(
            input_img_batch=self.img_batch,
            gtboxes_h_batch=None, gtboxes_r_batch=None)

        self.init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        self.restorer, self.restore_ckpt = self.det_net.get_restorer(model_path=self.model_path)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)

        self.restorer.restore(self.sess, self.restore_ckpt)
        print('AI Model Restored.'),


    def inference(self, image):
        inference_start = time.time()

        resized_img, det_boxes_h_, det_scores_h_, det_category_h_, \
        det_boxes_r_, det_scores_r_, det_category_r_ = \
            self.sess.run(
                [self.img_batch, self.det_boxes_h, self.det_scores_h, self.det_category_h,
                 self.det_boxes_r, self.det_scores_r, self.det_category_r],
                feed_dict={self.img_plac: image}
            )
        inference_end = time.time()
        inference_time = inference_end - inference_start
        print('R2CNN inference time : ', inference_time)

        result_image = draw_box_in_img_inference.draw_rotate_box_cv(np.squeeze(resized_img, 0),
                                                              boxes=det_boxes_r_,
                                                              labels=det_category_r_,
                                                              scores=det_scores_r_)
        #
        # result_image = draw_box_in_img_inference.draw_box_cv(np.squeeze(resized_img, 0),
        #                                                       boxes=det_boxes_h_,
        #                                                       labels=det_category_h_,
        #                                                       scores=det_scores_h_)


        return result_image, inference_time


if __name__ == '__main__':
    # images_dir = '/media/qisens/2tb1/python_projects/inference_pr/aws_ec2_inference_server_seg_slide/tmp/seochogu_office_blur'
    images_dir = '/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/train_bk/JPEGImages_new'
    output_dir = '/media/qisens/2tb1/python_projects/training_pr/R2CNN_Faster_RCNN_Tensorflow/inference_results/overfitting_result_'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model_path = "/media/qisens/2tb1/python_projects/training_pr/R2CNN_Faster_RCNN_Tensorflow/output/trained_weights_bk_166002/FasterRCNN_20180515_DOTA_v3/voc_166002model.ckpt"

    obj = R2CNN(model_path, gpu=0)

    infer_time_total = 0
    file_cnt = 0
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            img_path = os.path.join(root, file)
            output_img = os.path.join(output_dir, file)
            print(img_path)
            img = cv2.imread(img_path)
            img, inference_time = obj.inference(img)
            cv2.imwrite(output_img, img)
            infer_time_total += inference_time
            file_cnt += 1

    print("average inference time: {}".format(infer_time_total / file_cnt))

