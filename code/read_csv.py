#!/usr/bin/python
# -*- coding: utf-8 -*-
# ===========================================================
#  File Name: read_csv.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-15-2018
#  Last Modified: Thu Jan 10 17:48:10 2019
#
#  Usage: python read_csv.py
#  Description: Get final result
#
#  Copyright (C) 2018 Xu Zhang
#  All rights reserved.
#
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
# ===========================================================

# source code from
# https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/

from __future__ import print_function
import os
import cv2
import numpy as np
import pandas as pd
import csv


def err_quit(msg, exit_status=1):
    print(msg)
    exit(exit_status)


def load_csv(csv_fn, sep="|"):
    try:
        return pd.read_csv(csv_fn, sep, dtype=str)
    except IOError as ioerr:
        err_quit("{}. Aborting!".format(ioerr))


source_image_set = ['mouse-brain-P56-sice-3-21-atlas_cropped', 'P4_100033295_317_r', 'P7_100073790_64_r', 'P14_100016572_56_r_crop',
                    'WT-P7-Nissl-3', 'WT-P7-Nissl-5', 'WT-P7-Nissl-6']

experiment_setting = ['4_mask', '4_nrm_nal', '4_nrm', '4_ne', '4']


all_image_dict = {}
for source_1 in source_image_set:
    all_image_dict[source_1] = {}
    for setting_1 in experiment_setting:
        all_image_dict[source_1][setting_1] = []

all_setting_dict = {}
for setting_1 in experiment_setting:
    all_setting_dict[setting_1] = []

for source in source_image_set:
    image_dict = {}
    for source_1 in source_image_set:
        # if source_1 != source:
        image_dict[source_1] = []
    for setting in experiment_setting:
        score_list = load_csv(
            '../multi_region/result/csv_file/' +
            source +
            '_dice_region-{}.csv'.format(setting),
            ',')
        for column in score_list:
            if column in image_dict:
                tmp_list = []
                for score in score_list[column]:
                    tmp_list.append(float(score))
                image_dict[column].append(np.array(tmp_list))

                if column != source:
                    all_image_dict[column][setting].append(np.array(tmp_list))
                    all_setting_dict[setting].append(np.array(tmp_list))

    try:
        os.stat('../final_file/')
    except BaseException:
        os.makedirs('../final_file/')

    csv_file = csv.writer(
        open(
            '../final_file/{}_ave.csv'.format(source),
            'w'),
        delimiter=',')

    setting_dict = {}
    for setting_1 in experiment_setting:
        setting_dict[setting_1] = []

    csv_file.writerow(experiment_setting)
    for tmp_source in source_image_set:
        tem_list = [
            '{:.3f}'.format(
                np.mean(
                    image_dict[tmp_source][setting_idx])) for setting_idx,
            tmp_setting in enumerate(experiment_setting)]
        csv_file.writerow(tem_list)
        if tmp_source != source:
            for setting_idx, tmp_setting in enumerate(experiment_setting):
                setting_dict[tmp_setting].append(
                    np.mean(image_dict[tmp_source][setting_idx]))

    tem_list = [
        '{:.3f}'.format(
            np.mean(
                np.array(
                    setting_dict[tmp_setting]))) for tmp_setting in experiment_setting]
    csv_file.writerow(tem_list)

    csv_file = csv.writer(
        open(
            '../final_file/{}_std.csv'.format(source),
            'w'),
        delimiter=',')

    setting_dict = {}
    for setting_1 in experiment_setting:
        setting_dict[setting_1] = []

    csv_file.writerow(experiment_setting)
    for tmp_source in source_image_set:
        tem_list = [
            '{:.3f}'.format(
                np.std(
                    image_dict[tmp_source][setting_idx])) for setting_idx,
            tmp_setting in enumerate(experiment_setting)]
        csv_file.writerow(tem_list)
        if tmp_source != source:
            for setting_idx, tmp_setting in enumerate(experiment_setting):
                setting_dict[tmp_setting].append(
                    image_dict[tmp_source][setting_idx])

    tem_list = [
        '{:.3f}'.format(
            np.std(
                np.array(
                    setting_dict[tmp_setting]))) for tmp_setting in experiment_setting]
    csv_file.writerow(tem_list)

csv_file = csv.writer(open('../final_file/all_ave.csv', 'w'), delimiter=',')

csv_file.writerow(experiment_setting)
for tmp_source in source_image_set:
    tem_list = ['{:.3f}'.format(np.mean(np.array(all_image_dict[tmp_source][tmp_setting])))
                for setting_idx, tmp_setting in enumerate(experiment_setting)]
    csv_file.writerow(tem_list)

tem_list = [
    '{:.3f}'.format(
        np.mean(
            np.array(
                all_setting_dict[tmp_setting]))) for tmp_setting in experiment_setting]
csv_file.writerow(tem_list)

csv_file = csv.writer(open('../final_file/all_std.csv', 'w'), delimiter=',')

print(all_image_dict)
csv_file.writerow(experiment_setting)
for tmp_source in source_image_set:
    tem_list = ['{:.3f}'.format(np.std(np.array(all_image_dict[tmp_source][tmp_setting])))
                for setting_idx, tmp_setting in enumerate(experiment_setting)]
    csv_file.writerow(tem_list)

tem_list = [
    '{:.3f}'.format(
        np.std(
            np.array(
                all_setting_dict[tmp_setting]))) for tmp_setting in experiment_setting]
csv_file.writerow(tem_list)
