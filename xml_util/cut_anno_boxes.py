import os
import math
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
from xml.dom import minidom
import shutil
import cv2
import numpy as np


def is_exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="        ")


def make_dir_list(src_dir_folder, src_dir, separated_dir):
    src_anno_dir_list = []
    dst_anno_dir_list = []
    src_img_dir_list = []
    dst_img_dir_list = []
    for folder in src_dir_folder:
        src_anno_dir_list.append(os.path.join(src_dir, folder, 'annotations_new'))
        dst_anno_dir_list.append(os.path.join(separated_dir, folder, 'annotations_new'))
        src_img_dir_list.append(os.path.join(src_dir, folder, 'JPEGImages_new'))
        dst_img_dir_list.append(os.path.join(separated_dir, folder, 'JPEGImages_new'))
    print(src_anno_dir_list)

    return src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list


def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc
    yoff = yp - yc

    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return str(xc + pResx), str(yc + pResy)


def build_xml_arch(parsed_xml, current_img_file):
    label = ''
    cx = 0
    cy = 0
    width = 0
    height = 0
    angle = 0
    x0 = 0
    x1 = 0
    x2 = 0
    x3 = 0
    y0 = 0
    y1 = 0
    y2 = 0
    y3 = 0

    img = cv2.imread(current_img_file)
    img_height, img_width, img_channel = img.shape

    annotation = Element('annotation')

    folder = SubElement(annotation, 'folder')
    folder.text = parsed_xml['folder']

    filename = SubElement(annotation, 'filename')
    filename.text = parsed_xml['filename']

    path = SubElement(annotation, 'path')
    path.text = parsed_xml['path']

    source = SubElement(annotation, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'

    size = SubElement(annotation, 'size')
    width = SubElement(size, 'width')
    width.text = str(img_width)
    height = SubElement(size, 'height')
    height.text = str(img_height)
    depth = SubElement(size, 'depth')
    depth.text = str(img_channel)

    segmented = SubElement(annotation, 'segmented')
    segmented.text = '0'

    cnt = 0
    for obj in parsed_xml['objects']:
        # xml 태그가 robndbox 인 경우
        if not obj.find('robndbox') is None:
            label = obj.find('name').text
            cx = obj.find('./robndbox/cx').text
            cy = obj.find('./robndbox/cy').text
            width = obj.find('./robndbox/w').text
            height = obj.find('./robndbox/h').text
            angle = obj.find('./robndbox/angle').text
            try:
                x0 = obj.find('./robndbox/x0').text
                x1 = obj.find('./robndbox/x1').text
                x2 = obj.find('./robndbox/x2').text
                x3 = obj.find('./robndbox/x3').text
                y0 = obj.find('./robndbox/y0').text
                y1 = obj.find('./robndbox/y1').text
                y2 = obj.find('./robndbox/y2').text
                y3 = obj.find('./robndbox/y3').text
            except AttributeError:
                print("Found bndbox")
                cx = float(cx)
                cy = float(cy)
                w = float(width)
                h = float(height)
                angle = float(angle)

                x0, y0 = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
                x1, y1 = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
                x2, y2 = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
                x3, y3 = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)

                cx, cy, w, h, angle = str(cx), str(cy), str(w), str(h), str(angle)

                # str() >> 좌표들을 xml로 저장하기 위해선 text 로 바꿔야 하기 때문에 str 로 묶음
                # int() >> 좌표를 R2CNN 에선 정수형으로 변환하기 때문에 int 로 묶음
                # float() >> 실수형 좌표를 정수로 바꾸기 위해 float 으로 먼저 형변환을 해줌 >> 왜 해야하는거지
                x0, x1, x2, x3, y0, y1, y2, y3 = str(int(float(x0))), str(int(float(x1))), str(int(float(x2))), str(
                    int(float(x3))), str(int(float(y0))), str(int(float(y1))), str(int(float(y2))), str(int(float(y3)))

            object_tag = SubElement(annotation, 'object')
            type = SubElement(object_tag, 'type')
            type.text = 'robndbox'
            name = SubElement(object_tag, 'name')
            name.text = label
            pose = SubElement(object_tag, 'pose')
            pose.text = 'Unspecified'
            truncated = SubElement(object_tag, 'truncated')
            truncated.text = '0'
            difficult = SubElement(object_tag, 'difficult')
            difficult.text = '0'
            robndbox = SubElement(object_tag, 'robndbox')
            cx_tag = SubElement(robndbox, 'cx')
            cx_tag.text = cx
            cy_tag = SubElement(robndbox, 'cy')
            cy_tag.text = cy
            width_tag = SubElement(robndbox, 'w')
            width_tag.text = width
            height_tag = SubElement(robndbox, 'h')
            height_tag.text = height
            angle_tag = SubElement(robndbox, 'angle')
            angle_tag.text = angle
            x0_tag = SubElement(robndbox, 'x0')
            x0_tag.text = x0
            x1_tag = SubElement(robndbox, 'x1')
            x1_tag.text = x1
            x2_tag = SubElement(robndbox, 'x2')
            x2_tag.text = x2
            x3_tag = SubElement(robndbox, 'x3')
            x3_tag.text = x3
            y0_tag = SubElement(robndbox, 'y0')
            y0_tag.text = y0
            y1_tag = SubElement(robndbox, 'y1')
            y1_tag.text = y1
            y2_tag = SubElement(robndbox, 'y2')
            y2_tag.text = y2
            y3_tag = SubElement(robndbox, 'y3')
            y3_tag.text = y3

        # xml 태그가 bndbox 인 경우
        if not obj.find('bndbox') is None:
            # robndbox 를 다루기 때문에 중심, 세로, 높이를 계산해야 하지만 bndbox 는 회전각이 0도 이기때문에 max min 값을 그대로 사용가능
            # 중심, 세로, 높이를 중요시 하는 경우 0을 계산해서 수정.
            label = obj.find('name').text
            cx = str(0)
            cy = str(0)
            width = str(0)
            height = str(0)
            angle = str(0)
            x0 = obj.find('./bndbox/xmin').text
            x1 = obj.find('./bndbox/xmax').text
            x2 = obj.find('./bndbox/xmax').text
            x3 = obj.find('./bndbox/xmin').text
            y0 = obj.find('./bndbox/ymin').text
            y1 = obj.find('./bndbox/ymin').text
            y2 = obj.find('./bndbox/ymax').text
            y3 = obj.find('./bndbox/ymax').text

            object_tag = SubElement(annotation, 'object')
            type = SubElement(object_tag, 'type')
            type.text = 'robndbox'
            name = SubElement(object_tag, 'name')
            name.text = label
            pose = SubElement(object_tag, 'pose')
            pose.text = 'Unspecified'
            truncated = SubElement(object_tag, 'truncated')
            truncated.text = '0'
            difficult = SubElement(object_tag, 'difficult')
            difficult.text = '0'
            robndbox = SubElement(object_tag, 'robndbox')
            cx_tag = SubElement(robndbox, 'cx')
            cx_tag.text = cx
            cy_tag = SubElement(robndbox, 'cy')
            cy_tag.text = cy
            width_tag = SubElement(robndbox, 'w')
            width_tag.text = width
            height_tag = SubElement(robndbox, 'h')
            height_tag.text = height
            angle_tag = SubElement(robndbox, 'angle')
            angle_tag.text = angle
            x0_tag = SubElement(robndbox, 'x0')
            x0_tag.text = x0
            x1_tag = SubElement(robndbox, 'x1')
            x1_tag.text = x1
            x2_tag = SubElement(robndbox, 'x2')
            x2_tag.text = x2
            x3_tag = SubElement(robndbox, 'x3')
            x3_tag.text = x3
            y0_tag = SubElement(robndbox, 'y0')
            y0_tag.text = y0
            y1_tag = SubElement(robndbox, 'y1')
            y1_tag.text = y1
            y2_tag = SubElement(robndbox, 'y2')
            y2_tag.text = y2
            y3_tag = SubElement(robndbox, 'y3')
            y3_tag.text = y3
        filename_ = os.path.split(current_img_file)
        filename_ = os.path.splitext(filename_[-1])
        img_crop_bbox(img, filename_[0], int(x0),int(y0),int(x1),int(y1),int(x2),int(y2),int(x3),int(y3), cnt)
        cnt+=1

    result = ElementTree(annotation)

    # result = prettify(annotation)
    # print(result)
    # print('\n\n\n')
    return result


def img_crop_bbox(img, crop_image_path, x0, y0, x1, y1, x2, y2, x3, y3, cnt):
    img_height, img_width, _ = img.shape
    img = img + np.array([123.68, 116.779, 103.939])
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)
    # rect = ((x_c, y_c), (w, h), theta)
    # rect = cv2.boxPoints(rect)
    box_ls = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    rect = np.array(box_ls)
    rotated_bbox_coord = np.int0(rect)

    # save bbox cropped images
    ##########################################
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    points = np.array(rotated_bbox_coord)

    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
    rect = list(rect)
    if rect[0] < 0:
        rect[2] += rect[0]
        rect[0] = 0
    if rect[1] < 0:
        rect[3] += rect[1]
        rect[1] = 0
    if rect[0] > img_width:
        rect[0] = img_width
    if rect[1] > img_height:
        rect[1] = img_height

    width_y = rect[1] + rect[3]
    width_x = rect[0] + rect[2]
    if width_y > img_height:
        width_y = img_height
    if width_x > img_width:
        width_x = img_width

    cropped_image = res[rect[1]: width_y, rect[0]: width_x]

    crop_image_save = crop_image_path + "_" + "4points_" + str(x0) + "_" + str(y0) + "_" + str(x1) + "_" + str(y1) + "_" + str(x2) + "_" + str(y2) + "_" + str(x3) + "_" + str(y3) + "_" + ".png"
    new_image_path = os.path.join("/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/image_bndbox", crop_image_save)
    cv2.imwrite(new_image_path, cropped_image)




def xml_parser(path, file, current_img_file):
    xml_parse = {'folder': '', 'filename': '', 'path': '', 'objects': []}

    file_path = os.path.join(path, file)
    tree = ET.parse(file_path)

    obj_name_elements = tree.findall('./object')
    if len(obj_name_elements) == 0:  # object 가 하나도 없다면 return None
        return None
    xml_parse['objects'] = obj_name_elements

    bndbox_name_elements = tree.findall('*/bndbox')
    if len(bndbox_name_elements) == 0:
        pass

    robndbox_name_elements = tree.findall('*/robndbox')
    if len(robndbox_name_elements) == 0:
        pass

    xml_parse['folder'] = tree.getroot().find('folder').text
    xml_parse['filename'] = tree.getroot().find('filename').text
    xml_parse['path'] = tree.getroot().find('path').text

    new_xml_tree = build_xml_arch(xml_parse, current_img_file)

    return new_xml_tree


def walk_around_xml_files(src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list):
    err_file_cnt = 0
    for src_dir in src_anno_dir_list:
        index = src_anno_dir_list.index(src_dir)
        is_exist_dir(dst_img_dir_list[index])
        is_exist_dir(dst_anno_dir_list[index])
        for root, dirs, files in os.walk(src_dir):
            print('srcdir:', src_dir, " index : ", index)
            print('dstdir:', dst_anno_dir_list[index])
            for file in files:
                current_anno_file = os.path.join(str(root), file)
                current_img_file = os.path.join(src_img_dir_list[index], file[:-3] + 'png')

                dst_anno_file = os.path.join(dst_anno_dir_list[index], file)
                dst_img_file = os.path.join(dst_img_dir_list[index], file[:-3] + 'png')

                # To Do
                try:
                    print(file)
                    new_xml_tree = xml_parser(root, file, current_img_file)

                    if new_xml_tree is None:
                        shutil.copyfile(current_img_file, dst_img_file)
                        continue

                    # shutil.copyfile(current_anno_file, dst_anno_file)
                    new_xml_tree.write(dst_anno_file)
                    shutil.copyfile(current_img_file, dst_img_file)

                except FileNotFoundError:
                    print('ERROR : ', file)
                    err_file_cnt += 1
                    continue
                # To DO
    print("Error : ", err_file_cnt)


if __name__ == "__main__":
    # src_dir_folder = ['train', 'test', 'coa_origin']
    src_dir_folder = ['train', 'test']
    src_dir = '/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_clustering/'
    separated_dir = '/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/image_bndbox'

    src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list = make_dir_list(src_dir_folder, src_dir,
                                                                                             separated_dir)
    walk_around_xml_files(src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list)
