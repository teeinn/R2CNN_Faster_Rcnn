from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
import os
import shutil


xml_dir = "/media/qisens/2tb1/python_projects/training_pr/R2CNN_Faster_RCNN_Tensorflow/data/io/origin/robndbox/png/train/annotations"
for path, dirs, files in os.walk(xml_dir):
    for file in files:

        anno_path = os.path.join(path, file)

        doc = ET.parse(anno_path)
        root = doc.getroot()
        obj_tag = root.findall("object")

        if len(obj_tag) == 0:
            os.remove(anno_path)




