# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np

from libs.label_name_dict.label_dict import NAME_LABEL_MAP
from libs.configs import cfgs
from libs.box_utils import iou_rotate
from libs.box_utils import coordinate_convert
from help_utils import tools


def _write_voc_results_file(all_boxes, test_imgid_list, det_save_path):
  for cls, cls_ind in NAME_LABEL_MAP.items():
    if cls == 'back_ground':
      continue
    print('Writing {} VOC results file'.format(cls))

    with open(det_save_path, 'wt') as f:
      for im_ind, index in enumerate(test_imgid_list):
        dets = all_boxes[cls_ind][im_ind]
        if dets == []:
          continue
        # the VOCdevkit expects 1-based indices
        for k in range(dets.shape[0]):
          f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                  format(index, dets[k, -1],
                         dets[k, 0] + 1, dets[k, 1] + 1,
                         dets[k, 2] + 1, dets[k, 3] + 1, dets[k, 4] + 1))


def write_voc_results_file(all_boxes, test_imgid_list, det_save_dir):
  '''

  :param all_boxes: is a list. each item reprensent the detections of a img.
  the detections is a array. shape is [-1, 7]. [category, score, x, y, w, h, theta]
  Note that: if none detections in this img. that the detetions is : []

  :param test_imgid_list:
  :param det_save_path:
  :return:
  '''
  for cls, cls_id in NAME_LABEL_MAP.items():
    if cls == 'back_ground':
      continue
    print("Writing {} VOC resutls file".format(cls))

    tools.mkdir(det_save_dir)
    det_save_path = os.path.join(det_save_dir, "det_"+cls+".txt")
    with open(det_save_path, 'wt') as f:
      for index, img_name in enumerate(test_imgid_list):
        this_img_detections = all_boxes[index]

        this_cls_detections = this_img_detections[this_img_detections[:, 0] == cls_id]
        if this_cls_detections.shape[0] == 0:
          continue # this cls has none detections in this img
        for a_det in this_cls_detections:
          f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                  format(img_name, a_det[1],
                         a_det[2], a_det[3],
                         a_det[4], a_det[5], a_det[6]))  # that is [img_name, score, x, y, w, h, theta]


def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)

    try:
      try:
        bbox = obj.find('robndbox')
        #print('!!!RRR')
        #print(bbox)
      except:
        bbox = obj.find('bndbox')
        #print('@@@RRR')
        #print(bbox)

      rbox = [int(bbox.find('x0').text), int(bbox.find('y0').text), int(bbox.find('x1').text),
              int(bbox.find('y1').text), int(bbox.find('x2').text), int(bbox.find('y2').text),
              int(bbox.find('x3').text), int(bbox.find('y3').text)]
      rbox = np.array([rbox], np.float32)
      rbox = coordinate_convert.back_forward_convert(rbox, with_label=False)
      obj_struct['bbox'] = rbox
      objects.append(obj_struct)
    except:
      continue

  return objects


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def voc_eval(detpath, annopath, test_imgid_list, cls_name, ovthresh=0.5,
                 use_07_metric=False, use_diff=False):
  '''

  :param detpath:
  :param annopath:
  :param test_imgid_list: it 's a list that contains the img_name of test_imgs
  :param cls_name:
  :param ovthresh:
  :param use_07_metric:
  :param use_diff:
  :return:
  '''
  # 1. parse xml to get gtboxes

  # read list of images
  imagenames = test_imgid_list

  recs = {}
  for i, imagename in enumerate(imagenames):
    recs[imagename] = parse_rec(os.path.join(annopath, imagename+'.xml'))
    # if i % 100 == 0:
    #   print('Reading annotation for {:d}/{:d}'.format(
    #     i + 1, len(imagenames)))

  # 2. get gtboxes for this class.
  class_recs = {}
  num_pos = 0
  # if cls_name == 'person':
  #   print ("aaa")
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == cls_name]
    bbox = np.array([x['bbox'] for x in R])
    if use_diff:
      difficult = np.array([False for x in R]).astype(np.bool)
    else:
      difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    num_pos = num_pos + sum(~difficult)  # ignored the diffcult boxes
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det} # det means that gtboxes has already been detected

  # 3. read the detection file
  detfile = os.path.join(detpath, "det_"+cls_name+".txt")
  with open(detfile, 'r') as f:
    lines = f.readlines()

  # for a line. that is [img_name, confidence, xmin, ymin, xmax, ymax]
  splitlines = [x.strip().split(' ') for x in lines]  # a list that include a list
  image_ids = [x[0] for x in splitlines]  # img_id is img_name
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  nd = len(image_ids) # num of detections. That, a line is a det_box.
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]  #reorder the img_name

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]  # img_id is img_name
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        # ixmin = np.maximum(BBGT[:, 0], bb[0])
        # iymin = np.maximum(BBGT[:, 1], bb[1])
        # ixmax = np.minimum(BBGT[:, 2], bb[2])
        # iymax = np.minimum(BBGT[:, 3], bb[3])
        # iw = np.maximum(ixmax - ixmin + 1., 0.)
        # ih = np.maximum(iymax - iymin + 1., 0.)
        # inters = iw * ih
        #
        # # union
        # uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
        #        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
        #        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
        #
        # overlaps = inters / uni
        overlaps = []
        for i in range(len(BBGT)):
          overlap = iou_rotate.iou_rotate_calculate1(np.array([bb]),
                                                      BBGT[i],
                                                      use_gpu=False)[0]
          overlaps.append(overlap)
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1
          else:
            fp[d] = 1.
      else:
        fp[d] = 1.

  # 4. get recall, precison and AP
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(num_pos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)

  return rec, prec, ap


def get_values_from_dict(dict):
  value_list = list(dict.values())

  return value_list


def do_python_eval(test_imgid_list, test_annotation_path):
  import matplotlib.colors as colors
  import matplotlib.pyplot as plt

  mAP_dict = {}
  mPrecision_dict = {}
  mRecall_dict = {}
  for cls, index in NAME_LABEL_MAP.items():
    print(cls)
    if cls == 'back_ground':
      continue
    recall, precision, AP = voc_eval(detpath=cfgs.EVALUATE_R_DIR,
                                     test_imgid_list=test_imgid_list,
                                     cls_name=cls,
                                     annopath=test_annotation_path)

    Precision_cls = np.mean(precision) + np.finfo(np.float64).eps
    Recall_cls = np.mean(recall) + np.finfo(np.float64).eps
    print("{}_AP: {}".format(cls, AP))
    print("{}_mRecall: {}".format(cls, Recall_cls))
    print("{}_mPrecision: {}".format(cls, Precision_cls))

    mAP_dict[cls] = AP
    mPrecision_dict[cls] = Precision_cls
    mRecall_dict[cls] = Recall_cls

    c = colors.cnames.keys()
    c_dark = list(filter(lambda x: x.startswith('dark'), c))
    c = ['red', 'orange']
    plt.axis([0, 1.2, 0, 1])
    plt.plot(recall, precision, color=c_dark[index], label=cls)

  plt.legend(loc='upper right')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.savefig('./PR_R.png')

  print(mAP_dict, mRecall_dict, mPrecision_dict)
  total_mAP = np.mean(get_values_from_dict(mAP_dict))
  total_mRecall = np.mean(get_values_from_dict(mRecall_dict))
  total_mPrecision = np.mean(get_values_from_dict(mPrecision_dict))

  print("mAP_H is : {}".format(total_mAP))
  print("mRecall_H is : {}".format(total_mRecall))
  print("mPrecision_H is : {}".format(total_mPrecision))
  return total_mAP, total_mRecall, total_mPrecision, mAP_dict, mRecall_dict, mPrecision_dict

def get_label_np(test_imgid_list, annopath):
  imagenames = test_imgid_list
  class_ls = []
  for i, imagename in enumerate(imagenames):
    test_info = parse_rec(os.path.join(annopath, imagename + '.xml'))

    if test_info[0]['name'] == 'back_ground':
      class_ls.append(0)
    elif test_info[0]['name'] == 'flatroof':
      class_ls.append(1)
    elif test_info[0]['name'] == 'solarpanel_flat':
      class_ls.append(2)
    elif test_info[0]['name'] == 'solarpanel_slope':
      class_ls.append(3)
    elif test_info[0]['name'] == 'parkinglot':
      class_ls.append(4)
    elif test_info[0]['name'] == 'facility':
      class_ls.append(5)
    elif test_info[0]['name'] == 'rooftop':
      class_ls.append(6)
    elif test_info[0]['name'] == 'heliport_r':
      class_ls.append(7)
    elif test_info[0]['name'] == 'heliport_h':
      class_ls.append(8)

  return class_ls




def voc_evaluate_detections(all_boxes, test_imgid_list, test_annotation_path):
  '''

  :param all_boxes: is a list. each item reprensent the detections of a img.

  The detections is a array. shape is [-1, 6]. [category, score, xmin, ymin, xmax, ymax]
  Note that: if none detections in this img. that the detetions is : []
  :return:
  '''

  write_voc_results_file(all_boxes, test_imgid_list=test_imgid_list,
                         det_save_dir=cfgs.EVALUATE_R_DIR)
  mAP, recall, precision, total_AP, total_recall, total_precision = do_python_eval(test_imgid_list, test_annotation_path)
  # class_ls = get_label_np(test_imgid_list, test_annotation_path)
  return mAP, recall, precision, total_AP, total_recall, total_precision