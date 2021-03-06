# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
sys.path.append("../")
import tensorflow as tf
import time
import cv2
import pickle
import numpy as np
import argparse
import tensorflow.contrib.slim as slim

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from libs.val_libs import voc_eval, voc_eval_r
from libs.box_utils import draw_box_in_img
from libs.box_utils.coordinate_convert import forward_convert, back_forward_convert
from libs.label_name_dict.label_dict import LABEl_NAME_MAP, NAME_LABEL_MAP
from help_utils import tools


EVAL_INTERVAL = 250

def eval_with_plac(img_dir, det_net, num_imgs, image_ext, draw_imgs=False, test_annotation_path=None):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not GBR
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     is_resize=False)

    det_boxes_h, det_scores_h, det_category_h, \
    det_boxes_r, det_scores_r, det_category_r = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_h_batch=None, gtboxes_r_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    global_step_tensor = slim.get_or_create_global_step()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    last_checkpoint = None

    while True:
        restorer, restore_ckpt = det_net.get_restorer()

        with tf.Session(config=config) as sess:
            sess.run(init_op)
            sess.run(global_step_tensor.initializer)
            if not restorer is None:
                restorer.restore(sess, restore_ckpt)
                print('restore model')

            start_time = time.time()
            global_step = tf.train.global_step(sess, global_step_tensor)
            summary_path = os.path.join(cfgs.SUMMARY_PATH, cfgs.VERSION + "/eval_0")
            summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

            all_boxes_h = []
            all_boxes_r = []
            imgs = os.listdir(img_dir)
            for i, a_img_name in enumerate(imgs):
                a_img_name = a_img_name.split(image_ext)[0]

                raw_img = cv2.imread(os.path.join(img_dir,
                                                  a_img_name + image_ext))
                raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

                start = time.time()
                resized_img, det_boxes_h_, det_scores_h_, det_category_h_, \
                det_boxes_r_, det_scores_r_, det_category_r_ = \
                    sess.run(
                        [img_batch, det_boxes_h, det_scores_h, det_category_h,
                         det_boxes_r, det_scores_r, det_category_r],
                        feed_dict={img_plac: raw_img}
                    )
                end = time.time()
                # print("{} cost time : {} ".format(img_name, (end - start)))
                if draw_imgs:
                    det_detections_h = draw_box_in_img.draw_box_cv(np.squeeze(resized_img, 0),
                                                                   boxes=det_boxes_h_,
                                                                   labels=det_category_h_,
                                                                   scores=det_scores_h_)
                    det_detections_r = draw_box_in_img.draw_rotate_box_cv(np.squeeze(resized_img, 0),
                                                                          boxes=det_boxes_r_,
                                                                          labels=det_category_r_,
                                                                          scores=det_scores_r_)
                    save_dir = os.path.join(cfgs.TEST_SAVE_PATH, cfgs.VERSION)
                    tools.mkdir(save_dir)
                    cv2.imwrite(save_dir + '/' + a_img_name + '_h.jpg',
                                det_detections_h[:, :, ::-1])
                    cv2.imwrite(save_dir + '/' + a_img_name + '_r.jpg',
                                det_detections_r[:, :, ::-1])

                xmin, ymin, xmax, ymax = det_boxes_h_[:, 0], det_boxes_h_[:, 1], \
                                         det_boxes_h_[:, 2], det_boxes_h_[:, 3]

                if det_boxes_r_.shape[0] != 0:
                    resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
                    det_boxes_r_ = forward_convert(det_boxes_r_, False)
                    det_boxes_r_[:, 0::2] *= (raw_w / resized_w)
                    det_boxes_r_[:, 1::2] *= (raw_h / resized_h)
                    det_boxes_r_ = back_forward_convert(det_boxes_r_, False)

                x_c, y_c, w, h, theta = det_boxes_r_[:, 0], det_boxes_r_[:, 1], det_boxes_r_[:, 2], \
                                        det_boxes_r_[:, 3], det_boxes_r_[:, 4]

                xmin = xmin * raw_w / resized_w
                xmax = xmax * raw_w / resized_w
                ymin = ymin * raw_h / resized_h
                ymax = ymax * raw_h / resized_h

                boxes_h = np.transpose(np.stack([xmin, ymin, xmax, ymax]))
                boxes_r = np.transpose(np.stack([x_c, y_c, w, h, theta]))
                dets_h = np.hstack((det_category_h_.reshape(-1, 1),
                                    det_scores_h_.reshape(-1, 1),
                                    boxes_h))
                dets_r = np.hstack((det_category_r_.reshape(-1, 1),
                                    det_scores_r_.reshape(-1, 1),
                                    boxes_r))
                all_boxes_h.append(dets_h)
                all_boxes_r.append(dets_r)

                tools.view_bar('{} image cost {}s'.format(a_img_name, (end - start)), i + 1, len(imgs))

            imgs = os.listdir(img_dir)
            real_test_imgname_list = [i.split(image_ext)[0] for i in imgs]
            mAP_h, mRecall_h, mPrecision_h, Ap_dict_h, Recall_dict_h, Precision_dict_h = voc_eval.voc_evaluate_detections(all_boxes=all_boxes_h,
                                             test_imgid_list=real_test_imgname_list,
                                             test_annotation_path=test_annotation_path)

            mAP_r, mRecall_r, mPrecision_r, Ap_dict_r, Recall_dict_r, Precision_dict_r = voc_eval_r.voc_evaluate_detections(all_boxes=all_boxes_r,
                                               test_imgid_list=real_test_imgname_list,
                                               test_annotation_path=test_annotation_path)

            total_ls = {"mAP_h": mAP_h, "mAP_r": mAP_r, "mRecall_h": mRecall_h, "mRecall_r": mRecall_r, "mPrecision_h": mPrecision_h, "mPrecision_r": mPrecision_r}
            for (key, value) in total_ls.items():
                new_summary = tf.Summary()
                new_summary.value.add(tag='EVAL_Global/{}'.format(key), simple_value=value)
                summary_writer.add_summary(new_summary, global_step)

            label_list = list(NAME_LABEL_MAP.keys())
            label_list.remove('back_ground')

            for each_label in label_list:
                new_summary = tf.Summary()
                new_summary.value.add(tag='EVAL_Class_AP/{}_AP_h'.format(each_label),
                                      simple_value=Ap_dict_h[each_label])
                summary_writer.add_summary(new_summary, global_step)
                new_summary = tf.Summary()
                new_summary.value.add(tag='EVAL_Class_AP/{}_AP_r'.format(each_label),
                                      simple_value=Ap_dict_r[each_label])
                summary_writer.add_summary(new_summary, global_step)
                new_summary = tf.Summary()
                new_summary.value.add(tag='EVAL_Class_Recall/{}_Recall_h'.format(each_label),
                                      simple_value=Recall_dict_h[each_label])
                summary_writer.add_summary(new_summary, global_step)
                new_summary = tf.Summary()
                new_summary.value.add(tag='EVAL_Class_Recall/{}_Recall_r'.format(each_label),
                                      simple_value=Recall_dict_r[each_label])
                summary_writer.add_summary(new_summary, global_step)
                new_summary = tf.Summary()
                new_summary.value.add(tag='EVAL_Class_Precision/{}_Precision_h'.format(each_label),
                                      simple_value=Precision_dict_h[each_label])
                summary_writer.add_summary(new_summary, global_step)
                new_summary = tf.Summary()
                new_summary.value.add(tag='EVAL_Class_Precision/{}_Precision_r'.format(each_label),
                                      simple_value=Precision_dict_r[each_label])
                summary_writer.add_summary(new_summary, global_step)

        time_to_next_eval = start_time + EVAL_INTERVAL - time.time()
        if time_to_next_eval > 0:
            print("waiting for next evaluation...")
            time.sleep(time_to_next_eval)



def eval(num_imgs, img_dir, image_ext, test_annotation_path):

    faster_rcnn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=False)
    eval_with_plac(img_dir=img_dir, det_net=faster_rcnn, num_imgs=num_imgs, image_ext=image_ext, draw_imgs=True, test_annotation_path=test_annotation_path)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R2CNN network')
    parser.add_argument('--img_dir', dest='img_dir',
                        help='images path',
                        default='/mnt/USBB/gx/DOTA/DOTA_clip/val/images/', type=str)
    parser.add_argument('--image_ext', dest='image_ext',
                        help='image format',
                        default='.png', type=str)
    parser.add_argument('--test_annotation_path', dest='test_annotation_path',
                        help='test annotate path',
                        default=cfgs.TEST_ANNOTATION_PATH, type=str)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='0', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    eval(np.inf, args.img_dir, args.image_ext, args.test_annotation_path)

















