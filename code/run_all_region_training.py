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


# define gpu you want to use
gpu_set = ['0', '1']
# gpu_set = ['0', '1', '2', '3'] #if you want to use more

parameter_set = [\
    # cross entropy seems much better than iou in this experiment
    '--loss=cross_entropy --metric=iou-multi-region-with-background --use_background ',
    '--loss=cross_entropy --metric=iou-multi-region-with-background --use_background --no_ref_mask ',
    '--loss=cross_entropy --metric=iou-multi-region-with-background --use_background --no_ref_mask --no_alignment ',
]

number_gpu = len(gpu_set)
process_set = []
source = 'mouse-brain-P56-sice-3-21-atlas_cropped.jpg'

for run in range(1):
    command = 'python align_image.py --source={} '.format(source)
    subprocess.call(command, shell=True)

    for idx, parameter in enumerate(parameter_set):
        print('Test Parameter: {}'.format(parameter))

        command = 'python one_shot_training_all_region.py  --data_dir=../data/ --log_dir=../all_region/unet_log/ \
                --output_dir=../all_region/result/ --model_dir=../all_region/model/ \
                 {}  --augmentation_num=20 --num_epochs=500 \
                --gpu-id {} --idx={} '.format(parameter, gpu_set[idx % number_gpu], run)

        print(command)
        p = subprocess.Popen(shlex.split(command))
        process_set.append(p)

        if (idx + 1) % number_gpu == 0:
            print('Wait for process end')
            for sub_process in process_set:
                sub_process.wait()

            process_set = []
        time.sleep(10)

for sub_process in process_set:
    sub_process.wait()
