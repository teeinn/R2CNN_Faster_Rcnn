from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
import os

on_label_ls = ['flatroof', 'facility', 'rooftop']

xml_dir = "/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/test/annotations_original_all_change_to_flat"
for path, dirs, files in os.walk(xml_dir):
    for file in files:

        anno_path = os.path.join(path, file)

        with open(anno_path, 'a') as f:
            doc = ET.parse(anno_path)
            root = doc.getroot()
            obj_tag = root.findall("object")

            for obj in obj_tag:
                label = obj.find('name').text
                if label == on_label_ls[1] or label == on_label_ls[2]:
                    obj.find('name').text = on_label_ls[0]

            doc.write(anno_path)



