from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
import os

on_label_ls = ['flatroof']

xml_dir = "/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/test/annotations_original_only_flatroof"
for path, dirs, files in os.walk(xml_dir):
    for file in files:

        anno_path = os.path.join(path, file)

        with open(anno_path, 'a') as f:
            doc = ET.parse(anno_path)
            root = doc.getroot()
            obj_tag = root.findall("object")

            for obj in obj_tag:
                label = obj.find('name').text
                if not label in on_label_ls:
                    root.remove(obj)

            doc.write(anno_path)



