
#!/bin/sh
export DATASET_ROOT="/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/"
export DATASET_NAME='ROOF'
export IMAGES='train/JPEGImages_new'
export LABELS='train/annotations_new'
export TEST_IMAGES='test/JPEGImages_new'
export TEST_LABELS='test/annotations_new'
#export SAVE_DIR='/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/tfrecord/'
cd ./io

python convert_data_to_tfrecord_test.py --VOC_dir=$DATASET_ROOT --xml_dir=$TEST_LABELS --image_dir=$TEST_IMAGES --save_name='test' --img_format='.png' --dataset=$DATASET_NAME

python convert_data_to_tfrecord.py --VOC_dir=$DATASET_ROOT --xml_dir=$LABELS --image_dir=$IMAGES --save_name='train' --img_format='.png' --dataset=$DATASET_NAME


