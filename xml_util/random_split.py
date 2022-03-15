import os
import shutil
import random
import glob

train_data = "/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/train"
test_data = "/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/test"
img_path = "JPEGImages_new"
anno_path = "annotations_new"

img_ls = glob.glob(os.path.join(train_data, img_path)+"/*.png")
random_img_ls_for_test = random.sample(img_ls, 30)

anno_ls = []
for img in random_img_ls_for_test:
    anno_file = img.replace(img_path, anno_path).replace("png", "xml")
    if os.path.exists(anno_file):
        anno_ls.append(anno_file)

for idx, anno in enumerate(anno_ls):
    shutil.move(random_img_ls_for_test[idx], os.path.join(test_data, img_path))
    shutil.move(anno_ls[idx], os.path.join(test_data, anno_path))