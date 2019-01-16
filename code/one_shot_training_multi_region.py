#!/usr/bin/python
# -*- coding: utf-8 -*-
# ===========================================================
#  File Name: one_shot_training_multi_region.py
#  Author: Xu Zhang, Zhuowei Li, Columbia University
#  Creation Date: 09-15-2018
#  Last Modified: Tue Jan 15 10:28:12 2019
#
#  Usage: python one_shot_training_multi_region.py -h
#  Description: Segment hippocampus region in brain image.
#
#  Copyright (C) 2018 Xu Zhang, Zhuowei Li
#  All rights reserved.
#
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
# ===========================================================

from __future__ import print_function
import shutil
import os
import keras.backend as k
import utils
from one_shot_training_sequence import OneShotTrainingSequenceMultiRegion
from one_shot_validation_sequence import OneShotValidationSequenceMultiRegion
import train_utils
import U_net
import cv2
import numpy as np
import keras.callbacks
import csv
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--source_name',
    default='P4_100033295_317_r',
    help='Name of the source image')
parser.add_argument(
    '--data_dir',
    default='../data/',
    help='folder to image data')
parser.add_argument(
    '--log_dir',
    default='../multi_region/unet_log/',
    help='folder to output log')
parser.add_argument(
    '--model_dir',
    default='../multi_region/model/',
    help='folder to save model')
parser.add_argument(
    '--output_dir',
    default='../multi_region/result/',
    help='folder to draw result')

parser.add_argument('--unet_depth', default=6, type=int, help='depth for unet')
parser.add_argument('--start_filters', default=8, type=int,
                    help='starting hidden channel number')

parser.add_argument("--use_background", action="store_true",
                    help="use background as another channel for label, required for cross-entropy loss ")
parser.add_argument("--no_ref_mask", action="store_true",
                    help="do not use reference mask")
parser.add_argument("--no_alignment", action="store_true",
                    help="do not use alignment")
parser.add_argument("--no_expand", action="store_true",
                    help="do not use alignment")
parser.add_argument("--no_weight", action="store_true",
                    help="do not use alignment")
parser.add_argument("--update_mask", action="store_true",
                    help="do not use alignment")

parser.add_argument('--loss', default='iou-multi-region', type=str)
parser.add_argument('--metric', default='iou-multi-region', type=str)

parser.add_argument('--idx', default=0, type=int, help='for multi test')

parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument(
    '--learning_rate',
    default=0.001,
    type=float,
    help='learning rate')
parser.add_argument(
    '--batch_size',
    default=4,
    type=int,
    help='batch size for training')
parser.add_argument('--num_epochs', default=200, type=int,
                    help='number of epoch for training')
parser.add_argument(
    '--augmentation_num',
    default=20,
    type=int,
    help='number of augmentation images per epoch')
parser.add_argument(
    '--region_list',
    default="1-2-3-4",
    type=str,
    help='region used for training')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# hyper-parameters
image_size = (512, 512)
data_format = 'channels_last'
monitor = 'val_loss'

region_list = []
suffix = args.source_name + \
    '_{}'.format(args.loss) + '_region-{}'.format(args.region_list)

# add more experiment settings
if args.use_background:
    suffix = suffix + '_bg'
if args.no_ref_mask:
    suffix = suffix + '_nrm'
if args.no_alignment:
    suffix = suffix + '_nal'
if args.no_expand:
    suffix = suffix + '_ne'
if args.no_weight:
    suffix = suffix + '_nw'

for index in args.region_list.split('-'):
    region_list.append(int(index))

if args.use_background:
    channels = len(region_list) + 1
else:
    channels = len(region_list)

k.set_image_data_format(data_format)

training_image = cv2.imread('{}/img/all_img/{}.png'.format(args.data_dir,
                                                            args.source_name), cv2.IMREAD_UNCHANGED)
training_image = cv2.resize(
                    training_image,
                    image_size)

# n channel + 1 background channel
if args.use_background:
    training_mask = np.zeros(
        (image_size[0],
         image_size[1],
         len(region_list) + 1),
        dtype=np.uint8)
else:
    training_mask = np.zeros(
        (image_size[0],
         image_size[1],
         len(region_list)),
        dtype=np.uint8)

# no back ground ref channel
ref_mask = np.zeros(
    (image_size[0],
     image_size[1],
     len(region_list)),
    dtype=np.float32)

# get background mask
background_mask = np.zeros(
    (image_size[0],
     image_size[1]),
    dtype=np.uint8) + 255

for idx, region_idx in enumerate(region_list):
    tmp_mask = cv2.imread('{}/mask/all_masks/{}-{}.png'.format(args.data_dir,
                                                                            args.source_name, region_idx), cv2.IMREAD_GRAYSCALE)
    tmp_mask = cv2.resize(
                tmp_mask,
                image_size,
                cv2.INTER_NEAREST)
    tmp_mask = cv2.threshold(tmp_mask, 100, 255, cv2.THRESH_BINARY)[1]
    if args.use_background:
        training_mask[:, :, idx + 1] = tmp_mask
    else:
        training_mask[:, :, idx] = tmp_mask

    background_mask[tmp_mask > 100] = 0

    try:
        assert not args.update_mask
        os.stat('{}/ref_mask/{}-{}.png'.format(args.data_dir,
                                               args.source_name, region_idx))
        os.stat('{}/ref_mask/{}-expand-{}.png'.format(args.data_dir,
                                               args.source_name, region_idx))
        weighted_training_mask = cv2.imread('{}/ref_mask/{}-{}.png'.format(args.data_dir,
                                                                           args.source_name, region_idx), cv2.IMREAD_UNCHANGED)
        weighted_reference_mask = weighted_training_mask.astype(np.float32)
        expand_training_mask = cv2.imread('{}/ref_mask/{}-expand-{}.png'.format(args.data_dir,
                                                                           args.source_name, region_idx), cv2.IMREAD_UNCHANGED)
        expand_trainig_mask = expand_training_mask.astype(np.float32)
    except BaseException:
        expand_training_mask = utils.expandImage(
            tmp_mask.copy(), expand_pixel=25)
        weighted_reference_mask = utils.assignWeight2Mask(expand_training_mask)
        try:
            os.stat('{}/ref_mask/'.format(args.data_dir))
        except BaseException:
            os.makedirs('{}/ref_mask/'.format(args.data_dir))
        cv2.imwrite('{}/ref_mask/{}-{}.png'.format(args.data_dir, args.source_name, region_idx),
                    weighted_reference_mask.astype(np.uint8))
        cv2.imwrite('{}/ref_mask/{}-expand-{}.png'.format(args.data_dir, args.source_name, region_idx),
                    expand_training_mask.astype(np.uint8))

    if args.no_expand:
        ref_mask[:, :, idx] = tmp_mask
    else:
        if args.no_weight:
            ref_mask[:, :, idx] = expand_training_mask
        else:
            ref_mask[:, :, idx] = weighted_reference_mask

# set background mask
if args.use_background:
    training_mask[:, :, 0] = background_mask

train_sequence = OneShotTrainingSequenceMultiRegion(training_image, training_mask, ref_mask,
                                                    batch_size=args.batch_size, use_background=args.use_background,
                                                    no_ref_mask=args.no_ref_mask, augmentation_number=args.augmentation_num)

if args.no_alignment:
    test_image_dir = '{}/img/all_img/'.format(args.data_dir)
    test_mask_dir = '{}/mask/all_masks/'.format(args.data_dir)
else:
    test_image_dir = '{}/img/test_img_aligned/{}/'.format(
        args.data_dir, args.source_name)
    test_mask_dir = '{}/mask/all_masks_aligned/{}/'.format(
        args.data_dir, args.source_name)

# use training image as validation
# validation_sequence = OneShotTrainingSequenceMultiRegion(training_image, training_mask, ref_mask,\
#        batch_size=args.batch_size, use_background = args.use_background, \
#        no_ref_mask = args.no_ref_mask, augmentation_number = 4)

# use test image as validation
validation_sequence = OneShotValidationSequenceMultiRegion(test_image_dir, test_mask_dir,
                                                           ref_mask, region_list=region_list, batch_size=1,
                                                           use_background=args.use_background, no_ref_mask=args.no_ref_mask)

test_sequence = OneShotValidationSequenceMultiRegion(test_image_dir, test_mask_dir,
                                                     ref_mask, region_list=region_list, batch_size=1,
                                                     use_background=args.use_background, no_ref_mask=args.no_ref_mask)

# build model
model = U_net.UNET()
if 'cross' in args.loss:
    use_softmax = True
else:
    use_softmax = False

if args.use_background:
    input_channels = channels
else:
    input_channels = channels + 1

if args.no_ref_mask:
    input_channels = 1

u_net = model.BuildUnet((image_size[1], image_size[0], input_channels), args.start_filters, args.unet_depth,
                        dropout=False, _nchannel=channels, use_softmax=use_softmax)

train_utils.Compile(
    u_net,
    optimizer='Adam',
    lr=args.learning_rate,
    metric=args.metric,
    loss=args.loss)

try:
    os.stat('{}/{}/'.format(args.model_dir, suffix))
except BaseException:
    os.makedirs('{}/{}/'.format(args.model_dir, suffix))

model_check_point = keras.callbacks.ModelCheckpoint("{}/{}/best_model.h5".format(args.model_dir, suffix),
                                                    monitor=monitor, save_best_only=True, save_weights_only=True)

try:
    os.stat('{}/{}'.format(args.log_dir, suffix))
    shutil.rmtree('{}/{}'.format(args.log_dir, suffix))
except BaseException:
    pass

tensorboard = keras.callbacks.TensorBoard(
    log_dir='{}/{}'.format(args.log_dir, suffix), batch_size=args.batch_size)

u_net.fit_generator(
    train_sequence,
    validation_data=validation_sequence,
    validation_steps=len(validation_sequence),
    steps_per_epoch=args.augmentation_num / args.batch_size,
    epochs=args.num_epochs,
    verbose=1,
    callbacks=[model_check_point, tensorboard],
)

# for final test
try:
    os.stat('{}/{}/'.format(args.output_dir, suffix))
except BaseException:
    os.makedirs('{}/{}/'.format(args.output_dir, suffix))

u_net.load_weights("{}/{}/best_model.h5".format(args.model_dir, suffix))

try:
    os.stat('{}/csv_file/'.format(args.output_dir))
except BaseException:
    os.makedirs('{}/csv_file/'.format(args.output_dir))

if args.idx == 0:
    csv_file = csv.writer(open(
        '{}/csv_file/{}.csv'.format(args.output_dir, suffix, suffix), 'w'), delimiter=',')
else:
    csv_file = csv.writer(open(
        '{}/csv_file/{}.csv'.format(args.output_dir, suffix, suffix), 'a'), delimiter=',')

filename_list = []
all_iou_list = []

for i in range(len(test_sequence)):
    # get predict
    output = u_net.predict(test_sequence[i][0], batch_size=args.batch_size)
    # output = training_mask#[:,:,0]#output[0,:,:,:]
    output = output[0, :, :, :]

    if not args.use_background:
        background_channel = np.ones(
            (np.shape(output)[0], np.shape(output)[1], 1)) * 0.5
        output = np.concatenate((background_channel, output), axis=-1)
    index_output = np.argmax(output, axis=2)

    show_mask = utils.drawMultiRegionIndex(index_output)
    cv2.imwrite('{}/{}/{}.png'.format(args.output_dir, suffix,
                                      test_sequence.file_name_list[i]), show_mask)

    # transform image back to original size for iou calculation
    transform_matrix = np.load(
        "{}/transform_matrix/{}/{}.npy".format(
            args.data_dir,
            args.source_name,
            test_sequence.file_name_list[i]))

    tmp_iou_list = []
    new_index_output = None
    for idx, region_idx in enumerate(region_list):
        original_mask = cv2.imread("{}/mask/all_masks/{}-{}.png".format(args.data_dir,
                                                                        test_sequence.file_name_list[i], region_idx), cv2.IMREAD_GRAYSCALE)
        tmp_output = (index_output == (idx + 1))
        tmp_output = tmp_output.astype(np.uint8) * 255

        if not args.no_alignment:
            new_output = cv2.warpAffine(tmp_output, transform_matrix, (image_size[1], image_size[0]),
                                        flags=cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS + cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
        else:
            new_output = tmp_output.copy()

        new_output = cv2.resize(
            new_output, (original_mask.shape[1], original_mask.shape[0]))
        if new_index_output is None:
            new_index_output = np.zeros(
                (original_mask.shape[0], original_mask.shape[1]), dtype=np.uint8)
        new_index_output[new_output > 100] = idx + 1
        iou = train_utils.get_iou(original_mask > 100, new_output > 100)
        tmp_iou_list.append(iou)

    show_mask = utils.drawMultiRegionIndex(new_index_output)
    cv2.imwrite('{}/{}/{}_ori.png'.format(args.output_dir, suffix,
                                          test_sequence.file_name_list[i]), show_mask)
    filename_list.append(test_sequence.file_name_list[i])
    tmp_iou_list.append(np.mean(tmp_iou_list))
    all_iou_list.append(tmp_iou_list)

filename_list.append('Region Level Average')

if args.idx == 0:
    csv_file.writerow(filename_list)

# prepare data for csv
iou_dic = {}
for i in range(len(region_list) + 1):
    iou_dic['region' + str(i + 1)] = []

for iou_list in all_iou_list:
    for i, iou in enumerate(iou_list):
        iou_dic['region' + str(i + 1)].append(iou)

for i in range(len(region_list)):
    tem_list = iou_dic['region' + str(i + 1)]
    tem_list = ['{:.3f}'.format(x) for x in tem_list] + \
        ['{:.3f}'.format(np.mean(tem_list))]
    csv_file.writerow(tem_list)
