#!/bin/bash

conf_thresh=0.5
IoU_thresh=0.5

# model=./models/resnet50/resnet50-ssd-Epoch-199-Loss-2.105243469053699.pth
# model=./train_info/my_train/resnet50-ssd/resnet50-ssd-Epoch-199-Loss-3.0232349241933516.pth # --- trained in RGB


# # --- for vgg16-ssd - anchors101 (small sqare, rects)
# net=vgg16-ssd
# model=./models/vgg16_anchors101/vgg16-ssd-Epoch-199-Loss-2.0482876823794456.pth
# python eval_ssd.py \
# --net ${net} \
# --trained_model ${model} \
# --dataset_type voc \
# --dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --anchors 1,0,1 \
# --rectangles 2 --rectangles 2 --rectangles 2 --rectangles 2 --rectangles 2 --rectangles 2 \
# --num_anchors 3,3,3,3,3,3 \
# --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 \
# --eval_dir ./models/vgg16_anchors101/eval


# # --- for vgg16-ssd + fractional pooling w/ (conv + relu) 
# net=vgg16-ssd
# model=./models/vgg16_fracP_insertConvReLU/vgg16-ssd-Epoch-199-Loss-2.619078408518145.pth
# python eval_ssd.py \
# --net ${net} \
# --trained_model ${model} \
# --dataset_type voc \
# --dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --vgg_config 64,64,"F11",64,"F12",128,128,"F21",128,"F22",256,256,256,"F31",256,"F32",512,512,512,"F41",512,"F42",512,512,512 \
# --eval_dir ./models/vgg16_fracP_insertConvReLU/eval


# # --- for vgg16-ssd + fractional pooling w/o (conv + relu) 
# net=vgg16-ssd
# model=./models/vgg16_fracP_woConvReLU/vgg16-ssd-Epoch-199-Loss-2.9701730282075944.pth
# python eval_ssd.py \
# --net ${net} \
# --trained_model ${model} \
# --dataset_type voc \
# --dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --vgg_config 64,64,"F11","F12",128,128,"F21","F22",256,256,256,"F31","F32",512,512,512,"F41","F42",512,512,512 \
# --eval_dir ./models/vgg16_fracP_woConvReLU/eval


# --- for vgg16-ssd - sigmoid sampling
train=sw/035065
net=vgg16-ssd
model=./models/vgg16_sigmoid_thresh/${train}/vgg16-ssd-Epoch-200-Loss-2.7583961202252296.pth
python eval_ssd.py \
--net ${net} \
--trained_model ${model} \
--dataset_type voc \
--dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
--anchors 1,1,1 \
--rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
--num_anchors 4,6,6,6,4,4 \
--vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 \
--eval_dir ./models/vgg16_sigmoid_thresh/${train}/eval


# # --- for resnet50-ssd with 79.7mAP 
# net=resnet50-ssd
# model=./models/resnet50_ssd_voc_79.7.pth
# python eval_ssd.py \
# --net ${net} \
# --trained_model ${model} \
# --dataset_type voc \
# --dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --anchors 1,1,1 \
# --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 6,6,6,6,4,4 \
# --vgg_config 64,64,'M',128,128,'M',256,256,256,'C',512,512,512,'M',512,512,512 \
# --eval_dir ./models/res79/eval


# net=resnet50-ssd
# method=cosine
# model=./models/resnet50/${method}/resnet50-ssd-Epoch-199-Loss-2.920528462625319.pth
# python eval_ssd.py \
# --net ${net} \
# --trained_model ${model} \
# --dataset_type voc \
# --dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
# --anchors 1,1,1 \
# --rectangles 2 --rectangles 2 3 --rectangles 2 3 --rectangles 2 3 --rectangles 2 --rectangles 2 \
# --num_anchors 4,6,6,6,4,4 \
# --eval_dir ./models/resnet50/${method}/eval
