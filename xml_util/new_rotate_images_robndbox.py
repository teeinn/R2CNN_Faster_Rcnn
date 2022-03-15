import os
import math
import cv2
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
from xml.dom import minidom
import shutil
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
        src_anno_dir_list.append(os.path.join(src_dir, folder, 'annotations'))
        dst_anno_dir_list.append(os.path.join(separated_dir, folder, 'annotations'))
        src_img_dir_list.append(os.path.join(src_dir, folder, 'JPEGImages'))
        dst_img_dir_list.append(os.path.join(separated_dir, folder, 'JPEGImages'))
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


def rotate_box(bb, cx, cy, h, w, theta):
    new_bb = list(bb)
    # rad = np.deg2rad(theta)
    for i, coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        # Grab  the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        v = [coord[0],coord[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_bb[i] = (calculated[0],calculated[1])
    return new_bb


def build_xml_arch(REMOVE_OVER_RANGE_BBOX, parsed_xml, current_img_file, rotate_theta):
    label = ''
    width = 0
    height = 0
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    x0 = 0
    x1 = 0
    x2 = 0
    x3 = 0
    y0 = 0
    y1 = 0
    y2 = 0
    y3 = 0

    img_height, img_width, img_channel = cv2.imread(current_img_file).shape

    annotation = Element('annotation')

    folder = SubElement(annotation, 'folder')
    folder.text = parsed_xml['folder']

    filename = SubElement(annotation, 'filename')
    filename.text = str(parsed_xml['filename'])

    path = SubElement(annotation, 'path')
    path.text = parsed_xml['path']

    source = SubElement(annotation, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'

    size = SubElement(annotation, 'size')
    width = SubElement(size, 'width')
    width.text = str(int(img_width))
    height = SubElement(size, 'height')
    height.text = str(int(img_height))
    depth = SubElement(size, 'depth')
    depth.text = str(img_channel)

    segmented = SubElement(annotation, 'segmented')
    segmented.text = '0'

    img_center_x = int((img_width) / 2)
    img_center_y = int((img_height) / 2)

    for obj in parsed_xml['objects']:
        # xml 태그가 robndbox 인 경우
        if not obj.find('robndbox') is None:
            label = obj.find('name').text
            cx = obj.find('./robndbox/cx').text
            cy = obj.find('./robndbox/cy').text
            width = obj.find('./robndbox/w').text
            height = obj.find('./robndbox/h').text
            angle = obj.find('./robndbox/angle').text
            if angle is None:
                angle = 0
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
                print("누락")
                cx = float(cx)
                cy = float(cy)
                w = float(width)
                h = float(height)
                angle = float(angle)
                if angle is None:
                    angle = 0

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
            try:
                x0_, y0_, x1_, y1_, x2_, y2_, x3_, y3_ = float(x0), float(y0), float(x1), float(y1), float(x2), float(y2), float(x3), float(y3)
                bb, cx, cy, w, h = [(x0_, y0_), (x1_, y1_), (x2_, y2_), (x3_, y3_)], float(cx), float(cy), float(width), float(height)
            except:
                bb, cx, cy, angle, width, height = [(x0_, y0_), (x1_, y1_), (x2_, y2_), (x3_, y3_)], cx, cy, angle, width, height

            new_bbox = rotate_box(bb, img_center_x, img_center_y, img_height, img_width, rotate_theta)
            x0, y0, x1, y1, x2, y2, x3, y3 = int(new_bbox[0][0]), int(new_bbox[0][1]), int(new_bbox[1][0]), int(
                new_bbox[1][1]), int(new_bbox[2][0]), int(new_bbox[2][1]), int(new_bbox[3][0]), int(new_bbox[3][1])

            img = cv2.imread(current_img_file)
            img = rotate_image(current_img_file, rotate_theta)
            cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), thickness=2)
            cv2.line(img, (x3, y3), (x0, y0), (0, 255, 0), thickness=2)

            if REMOVE_OVER_RANGE_BBOX:
                if max(x0, x1, x2, x3) >= img_width or max(y0, y1, y2, y3) >= img_height or \
                        min(x0, x1, x2, x3) < 0 or min(y0, y1, y2, y3) < 0:
                    continue

            cx = str((max(x0, x1, x2, x3) + min(x0, x1, x2, x3)) / 2)
            cy = str((max(y0, y1, y2, y3) + min(y0, y1, y2, y3)) / 2)
            x0, y0, x1, y1, x2, y2, x3, y3 = str(x0), str(y0), str(x1), str(y1), str(x2), str(y2), str(x3), str(y3)
            try:
                angle = str(-math.radians(rotate_theta) + float(angle))
            except TypeError:
                print('here')

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
            cx = 0
            cy = 0
            width = 0
            height = 0
            angle = 0
            x0 = obj.find('./bndbox/xmin').text
            x1 = obj.find('./bndbox/xmax').text
            x2 = obj.find('./bndbox/xmax').text
            x3 = obj.find('./bndbox/xmin').text
            y0 = obj.find('./bndbox/ymin').text
            y1 = obj.find('./bndbox/ymin').text
            y2 = obj.find('./bndbox/ymax').text
            y3 = obj.find('./bndbox/ymax').text

            try:
                x0_, y0_, x1_, y1_, x2_, y2_, x3_, y3_ = float(x0), float(y0), float(x1), float(y1), float(x2), float(
                    y2), float(x3), float(y3)
                bb, cx, cy, w, h = [(x0_, y0_), (x1_, y1_), (x2_, y2_), (x3_, y3_)], float(cx), float(cy), float(
                    width), float(height)
            except:
                bb, cx, cy, angle, width, height = [(x0_, y0_), (x1_, y1_), (x2_, y2_), (x3_, y3_)], cx, cy, angle, width, height

            new_bbox = rotate_box(bb, img_width // 2, img_height // 2, img_height, img_width, rotate_theta)
            x0, y0, x1, y1, x2, y2, x3, y3 = int(new_bbox[0][0]), int(new_bbox[0][1]), int(new_bbox[1][0]), int(
                new_bbox[1][1]), int(new_bbox[2][0]), int(new_bbox[2][1]), int(new_bbox[3][0]), int(new_bbox[3][1])

            img = cv2.imread(current_img_file)
            img = rotate_image(current_img_file, rotate_theta)
            cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), thickness=2)
            cv2.line(img, (x3, y3), (x0, y0), (0, 255, 0), thickness=2)

            cx = str((max(x0, x1, x2, x3) + min(x0, x1, x2, x3)) / 2)
            cy = str((max(y0, y1, y2, y3) + min(y0, y1, y2, y3)) / 2)

            x0, y0, x1, y1, x2, y2, x3, y3 = str(x0), str(y0), str(x1), str(y1), str(x2), str(y2), str(x3), str(y3)

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

    result = ElementTree(annotation)
    return result


def xml_parser(REMOVE_OVER_RANGE_BBOX, root, file, current_img_file, rotate_theta):
    xml_parse = {'folder': '', 'filename': '', 'path': '', 'objects': []}

    file_path = os.path.join(root, file)
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

    new_xml_tree = build_xml_arch(REMOVE_OVER_RANGE_BBOX, xml_parse, current_img_file, rotate_theta)

    return new_xml_tree


def rotate_image(src_image_path, rotate_theta):
    img = cv2.imread(src_image_path)
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), rotate_theta, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    img = cv2.warpAffine(img, M, (nW, nH))

    return img


def walk_around_xml_files(REMOVE_OVER_RANGE_BBOX, rotate_theta, src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list):
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

                dst_anno_file = os.path.join(dst_anno_dir_list[index], file[:-4] + '_' + 'fit_rotate_' + str(rotate_theta) + '.xml')
                dst_img_file = os.path.join(dst_img_dir_list[index], file[:-4] + '_' + 'fit_rotate_' + str(rotate_theta) + '.png')

                # To Do
                try:
                    print(file)
                    new_xml_tree = xml_parser(REMOVE_OVER_RANGE_BBOX, root, file, current_img_file, rotate_theta)

                    # annotation 파일이 없다면 이미지만 resize
                    if new_xml_tree is None:
                        resized_img = rotate_image(current_img_file, rotate_theta)
                        cv2.imwrite(dst_img_file, resized_img)
                        continue

                    # annotation 파일이 있다면 좌표값과 이미지 함께 resize
                    new_xml_tree.write(dst_anno_file)  # resize 좌표가 있는 xml 파일 저장
                    resized_img = rotate_image(current_img_file, rotate_theta)  # 이미지 resize
                    cv2.imwrite(dst_img_file, resized_img)

                except FileNotFoundError:
                    print('ERROR : ', file)
                    err_file_cnt += 1
                    continue
                # To DO
    print("Error : ", err_file_cnt)


# Image Argumentation, 이미지를 회전시키며 회전각 만큼 bbox도 회전시킴
# Rotate RCNN 전용 코드이므로 bbox 또한 회전함
if __name__ == "__main__":
    # # 이미지 범위를 벗어난 bbox 는 삭제됨
    # REMOVE_OVER_RANGE_BBOX = False
    # src_dir_folder = ['train']
    # src_dir = "/media/qisens/2tb1/python_projects/training_pr/R2CNN_Faster_RCNN_Tensorflow/data/io/output_new_bk/train/blur_resize0.8"
    # separated_dir = "/media/qisens/2tb1/python_projects/training_pr/R2CNN_Faster_RCNN_Tensorflow/data/io/output_new_bk/train/blur/blur_rotate60"
    #
    # rotate_theta = 60
    #
    # src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list = make_dir_list(src_dir_folder, src_dir,
    #                                                                                          separated_dir)
    # walk_around_xml_files(REMOVE_OVER_RANGE_BBOX, rotate_theta, src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list)

    # 이미지 범위를 벗어난 bbox 는 삭제됨
    REMOVE_OVER_RANGE_BBOX = False
    src_dir_folder = ['train']
    src_dir = "/media/qisens/2tb1/python_projects/training_pr/R2CNN_Faster_RCNN_Tensorflow/data/io/train_output_sharpen_blur_nesicnaverstyle_naverdata(origin_sharp_blur_resize_rotation)/test"
    separated_dir = "/media/qisens/2tb1/python_projects/training_pr/R2CNN_Faster_RCNN_Tensorflow/data/io/train_output_sharpen_blur_nesicnaverstyle_naverdata(origin_sharp_blur_resize_rotation)/testtest"
    filename_ls = [dirs for path, dirs, files in os.walk(src_dir)][0]
    rotate_theta = [30, 60]

    for dir_name in filename_ls:
        for theta in rotate_theta:
            new_src_dir = os.path.join(src_dir, dir_name)
            new_separated_dir = separated_dir + dir_name + "_rotate" + str(theta)


            src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list = make_dir_list(src_dir_folder, new_src_dir,
                                                                                                     new_separated_dir)
            walk_around_xml_files(REMOVE_OVER_RANGE_BBOX, theta, src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list)

