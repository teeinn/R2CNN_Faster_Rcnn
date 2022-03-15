import os


cnt = 0
cnt_ = 0
xml_dir = "/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/train/annotations_new"
img_dir = "/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/train/JPEGImages_new"
for path, dirs, files in os.walk(xml_dir):
    for file in files:

        anno_path = os.path.join(path, file)
        img_path = os.path.join(img_dir, file.replace("xml", "png"))

        if not os.path.exists(img_path):
            print(img_path)
            cnt += 1
            print("cnt : {}".format(cnt))
        else:
            print("it is existed")
            cnt_ += 1
            print("cnt_ : {}".format(cnt_))