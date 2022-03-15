import os
import shutil
anno_path = '/media/qisens/2tb1/goodroof_solarpanel_parkinglot_rooftop_facility_rotation/test/annotations'
new_img_path = '/media/qisens/2tb1/goodroof_solarpanel_parkinglot_rooftop_facility_rotation/test/JPEGImages_proved'
for path, dirs, files in os.walk(anno_path):
    for file in files:
        anno_path = os.path.join(path, file)
        img_path = anno_path.replace("annotations", "JPEGImages").replace('.xml', '.png')
        if os.path.isfile(img_path):
            shutil.copyfile(img_path, os.path.join(new_img_path, file.replace('.xml', '.png')))