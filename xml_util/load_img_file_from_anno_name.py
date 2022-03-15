import os
import shutil

xml_dir = "/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/train/annos_new"
img_dir = "/home/qisens/2020.3~/roof_detection_data/training/images"
new_img_dir = "/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/train/images_new"

for path, dirs, files in os.walk(img_dir):
    for file in files:

        img_path = os.path.join(path, file)
        anno_path = os.path.join(xml_dir, file.replace("png", "xml"))

        if os.path.exists(anno_path):
            shutil.copy(img_path, new_img_dir)