from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
import os

cluster_ls = ['cluster0', 'cluster1']
cnt = 0
xml_dir = "/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_clustering/test/annotations_new"


for cluster_name in cluster_ls:
    cluster_dir = "/media/qisens/2tb1/python_projects/training_pr/image-clustering/output/" + cluster_name
    for path, dirs, files in os.walk(xml_dir):
        for file in files:

            xml_filename = os.path.splitext(file)
            anno_path = os.path.join(path, file)

            with open(anno_path, 'a') as f:
                doc = ET.parse(anno_path)
                root = doc.getroot()
                obj_tag = root.findall("object")

                for obj in obj_tag:
                    label = obj.find('name').text
                    x0 = obj.find('./robndbox/x0').text
                    x1 = obj.find('./robndbox/x1').text
                    x2 = obj.find('./robndbox/x2').text
                    x3 = obj.find('./robndbox/x3').text
                    y0 = obj.find('./robndbox/y0').text
                    y1 = obj.find('./robndbox/y1').text
                    y2 = obj.find('./robndbox/y2').text
                    y3 = obj.find('./robndbox/y3').text

                    for cls_path, cls_dirs, cls_files in os.walk(cluster_dir):
                        for cls_file in cls_files:
                            cls_filename = cls_file.split('_4points_')
                            points_ls = cls_filename[-1].split('_')
                            if xml_filename[0] == cls_filename[0]:
                                if x0 == points_ls[0] and y0 == points_ls[1] and x1 == points_ls[2] and y1 == points_ls[3] and x2 == points_ls[4] and y2 == points_ls[5] and x3 == points_ls[6] and y3 == points_ls[7]:
                                    obj.find('name').text = cluster_name
                                    doc.write(anno_path)
                                    print(cls_filename[0])
                                    cnt += 1

print("cnt: ", cnt)



