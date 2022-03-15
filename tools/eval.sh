#!/bin/sh

#export IMAGES=$(pwd)/coa_origin/JPEGImages/

#export LABELS=$(pwd)/coa_origin/annotations/


export IMAGES=/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/test/JPEGImages_new/

export LABELS=/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/test/annotations_new/

python eval.py --img_dir=$IMAGES --image_ext='.png' --test_annotation_path=$LABELS --gpu='0'





