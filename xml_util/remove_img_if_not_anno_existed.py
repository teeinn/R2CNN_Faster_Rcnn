import os



xml_dir = "/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/train/annos_new"
img_dir = "/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/train/images_new"

for path, dirs, files in os.walk(img_dir):
    for file in files:

        img_path = os.path.join(path, file)
        anno_path = os.path.join(img_dir, file.replace("png", "xml"))

        if not os.path.exists(anno_path):
            print("remove {} file".format(img_path))
            os.remove(img_path)


