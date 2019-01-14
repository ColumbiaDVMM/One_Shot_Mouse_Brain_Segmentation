"""
Run multiple parameter with multiple GPUs and one python script 
Usage: python run_all.py

Author: Xu Zhang
Email: xu.zhang@columbia.edu.cn
"""

#! /usr/bin/env python2

import numpy as np
import scipy.io as sio
import time
import os
import sys
import subprocess
import shlex
import argparse


####################################################################
# Parse command line
####################################################################
def usage():
    print >> sys.stderr 
    sys.exit(1)


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

#define gpu you want to use
gpu_set = ['0', '1']
#gpu_set = ['0', '1', '2', '3'] #if you want to use more

parameter_set = [\
        # one single region test
        #'--region_list=4 --loss=dice --metric=iou-multi-region ',
        '--region_list=4 --loss=dice --metric=iou-multi-region --no_expand',
        #'--region_list=4 --loss=dice --metric=iou-multi-region --no_alignment --no_ref_mask ',
        '--region_list=4 --loss=dice --metric=iou-multi-region --no_ref_mask ',

        # 4 region test, feel free to change the region list
        #'--region_list=1-2-3-4 --loss=iou-multi-region --metric=iou-multi-region-with-background --use_background --no_alignment ',
        #'--region_list=1-2-3-4 --loss=iou-multi-region --metric=iou-multi-region-with-background --use_background --no_ref_mask ',
        #'--region_list=1-2-3-4 --loss=iou-multi-region --metric=iou-multi-region-with-background --use_background --no_alignment --no_ref_mask ',
        #'--region_list=1-2-3-4 --loss=iou-multi-region --metric=iou-multi-region-with-background --use_background ',

        #'--region_list=1-2-3-4 --loss=iou-multi-region --metric=iou-multi-region --no_alignment ',
        #'--region_list=1-2-3-4 --loss=iou-multi-region --metric=iou-multi-region --no_ref_mask ',
        #'--region_list=1-2-3-4 --loss=iou-multi-region --metric=iou-multi-region ',
        #'--region_list=1-2-3-4 --loss=iou-multi-region --metric=iou-multi-region-with-background --use_background --no_ref_mask ',
        #'--region_list=1-2-3-4 --loss=iou-multi-region --metric=iou-multi-region-with-background --use_background ',
        ]

number_gpu = len(gpu_set)
process_set = []

#run 2 times and record all the results. Run 10 times for final test.
source_list = ['mouse-brain-P56-sice-3-21-atlas_cropped.jpg', 'P4_100033295_317_r.jpg', 'P7_100073790_64_r.jpg', 'P14_100016572_56_r_crop.jpg',
        'WT-P7-Nissl-3.jpeg', 'WT-P7-Nissl-5.jpeg', 'WT-P7-Nissl-6.jpeg']

for source_idx, source in enumerate(source_list):
    command = 'python align_image.py --source={} '.format(source)
    source_name = source.split('.')[0]
    print(command)
    subprocess.call(command,shell=True)
    for run in range(10):
        source_name = source.split('.')[0]
        for idx, parameter in enumerate(parameter_set):
            print('Test Parameter: {}'.format(parameter))
            command = 'python one_shot_training_multi_region.py  --data_dir=../data/ --source_name={} --log_dir=../multi_region/unet_log/ \
                    --output_dir=../multi_region/result/ --model_dir=../multi_region/model/ \
                     {}  --augmentation_num=20 --num_epochs=200 \
                    --gpu-id {} --idx={} '.format(source_name, parameter, gpu_set[source_idx%number_gpu], run)
    
            print(command)
            p = subprocess.Popen(shlex.split(command))
            process_set.append(p)
            
            if (source_idx+1)%number_gpu == 0:
                print('Wait for process end')
                for sub_process in process_set:
                    sub_process.wait()
            
                process_set = []
            time.sleep(10)

    for sub_process in process_set:
        sub_process.wait()

