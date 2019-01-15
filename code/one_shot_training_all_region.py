#!/usr/bin/python
# -*- coding: utf-8 -*-
# ===========================================================
#  File Name: one_shot_training_all_region.py
#  Author: Xu Zhang, Zhuowei Li, Columbia University
#  Creation Date: 09-15-2018
#  Last Modified: Tue Jan 15 09:56:04 2019
#
#  Usage: python one_shot_training_all_region.py -h
#  Description: Segment all 95 regions for brain image.
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
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--source_name',
    default='mouse-brain-P56-sice-3-21-atlas_cropped',
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
                    help="Update reference mask or not")
parser.add_argument("--no_ref_mask", action="store_true",
                    help="Update reference mask or not")
parser.add_argument("--no_alignment", action="store_true",
                    help="Update reference mask or not")
parser.add_argument("--all_zeros", action="store_true",
                    help="Use all black mask as baseline")

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
    help='number of augmentated images per epoch')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# hyper-parameters
image_size = (512, 512)
data_format = 'channels_last'
# save model with best validation loss
monitor = 'val_loss'

region_list = []
suffix = args.source_name + \
    '_{}'.format(args.loss) + '_region-all' + \
    '_lr_{:1.1e}'.format(args.learning_rate)

# add more experiment settings
if args.use_background:
    suffix = suffix + '_bg'
if args.no_ref_mask:
    suffix = suffix + '_nrm'
if args.no_alignment:
    suffix = suffix + '_nal'
if args.no_expand:
    suffix = suffix + '_ne'
if args.all_zeros:
    suffix = suffix + '_az'


for index in range(1, 96):  # all regions
    region_list.append(int(index))

if args.use_background:
    channels = len(region_list) + 1
else:
    channels = len(region_list)

k.set_image_data_format(data_format)

training_image = cv2.imread('{}/img/training/{}.png'.format(args.data_dir,
                                                            args.source_name), cv2.IMREAD_UNCHANGED)

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
    tmp_mask = cv2.imread('{}/mask/95_masks/{}-{}.png'.format(args.data_dir,
                                                              args.source_name, region_idx), cv2.IMREAD_UNCHANGED)
    tmp_mask = cv2.threshold(tmp_mask, 100, 255, cv2.THRESH_BINARY)[1]

    if args.use_background:
        training_mask[:, :, idx + 1] = tmp_mask
    else:
        training_mask[:, :, idx] = tmp_mask
    background_mask[tmp_mask > 100] = 0
    ref_mask[:, :, idx] = tmp_mask

    if args.all_zeros:
        ref_mask[:, :, idx] = 0

if args.use_background:
    training_mask[:, :, 0] = background_mask

train_sequence = OneShotTrainingSequenceMultiRegion(training_image, training_mask, ref_mask,
                                                    batch_size=args.batch_size, use_background=args.use_background,
                                                    no_ref_mask=args.no_ref_mask)

validation_image_dir = '{}/img/validation_img/'.format(args.data_dir)
if args.no_alignment:
    test_image_dir = '{}/img/test_img/'.format(args.data_dir)
else:
    test_image_dir = '{}/img/test_img_aligned/{}/'.format(args.data_dir, args.source_name)

# use training img as validation
validation_sequence = OneShotValidationSequenceMultiRegion(validation_image_dir, '{}/mask/95_masks/'.format(args.data_dir),
                                                           ref_mask, region_list=region_list, batch_size=1,
                                                           use_background=args.use_background, no_ref_mask=args.no_ref_mask)

test_sequence = OneShotValidationSequenceMultiRegion(test_image_dir, None,
                                                     ref_mask, region_list=region_list, batch_size=1,
                                                     use_background=args.use_background, no_ref_mask=args.no_ref_mask, no_mask=True)

# model
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

filename_list = []
all_iou_list = []

for i in range(len(test_sequence)):
    # get predict
    output = u_net.predict(test_sequence[i], batch_size=args.batch_size)
    output = output[0, :, :, :]

    # if no backgound, add one dummy channel for drawing
    if not args.use_background:
        background_channel = np.ones(
            (np.shape(output)[0], np.shape(output)[1], 1)) * 0.5
        output = np.concatenate((background_channel, output), axis=-1)

    index_output = np.argmax(output, axis=2)
    show_mask = utils.drawMultiRegionIndex(index_output)
    cv2.imwrite('{}/{}/{}.png'.format(args.output_dir, suffix,
                                      test_sequence.file_name_list[i]), show_mask)

    transform_matrix = np.load(
        "{}/transform_matrix/{}/{}.npy".format(args.data_dir, args.source_name, test_sequence.file_name_list[i]))

    new_index_output = None
    for idx, region_idx in enumerate(region_list):
        tmp_output = (index_output == (idx + 1))
        tmp_output = tmp_output.astype(np.uint8) * 255

        if not args.no_alignment:
            new_output = cv2.warpAffine(tmp_output, transform_matrix, (image_size[1], image_size[0]),
                                        flags=cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS + cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
        else:
            new_output = tmp_output.copy()

        original_img = cv2.imread("{}/img/test_img/{}.png".format(args.data_dir,
                                                                  test_sequence.file_name_list[i]), cv2.IMREAD_GRAYSCALE)

        new_output = cv2.resize(
            new_output, (original_img.shape[1], original_img.shape[0]))
        if new_index_output is None:
            new_index_output = np.zeros(
                (original_img.shape[0], original_img.shape[1]), dtype=np.uint8)
        new_index_output[new_output > 100] = idx + 1

    show_mask = utils.drawMultiRegionIndex(new_index_output)
    cv2.imwrite('{}/{}/{}_ori.png'.format(args.output_dir, suffix,
                                          test_sequence.file_name_list[i]), show_mask)
