import numpy as np
import cv2
#from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import Sequence
from imgaug import augmenters as iaa
import glob
import utils


class OneShotValidationSequenceMultiRegion(Sequence):
    def __init__(self, image_dir, mask_dir, ref_mask, region_list=[4], batch_size=1, use_background=True,
                 no_ref_mask=False, no_mask=False, augmentation_number=200, image_width=512, image_height=512):

        self.region_list = region_list
        self.ref_mask = ref_mask
        self.image_width = image_width
        self.image_height = image_height
        self.use_background = use_background
        self.no_ref_mask = no_ref_mask
        self.no_mask = no_mask
        self.image, self.mask, self.file_name_list = \
            self.load_image_and_mask(image_dir, mask_dir)
        self.batch_size = batch_size
        self.channel_number = len(region_list) + 1

    def load_image_and_mask(self, image_dir, mask_dir):
        image_list = []
        mask_list = []
        file_name_list = []
        for img_name in glob.glob(image_dir + '/*.png'):
            pure_name = img_name.split('/')[-1]
            pure_name = pure_name.split('.')[0]
            file_name_list.append(pure_name)
            im_gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.resize(
                im_gray, (self.image_width, self.image_height))
            # aug_image = cv2.equalizeHist(im_gray)
            aug_image = im_gray
            image_list.append(aug_image)
            if not self.no_mask:
                if self.use_background:
                    multi_channel_mask = np.zeros(
                        (self.image_height, self.image_width, len(
                            self.region_list) + 1), dtype=np.uint8)
                    index_mask = np.zeros(
                        (self.image_width, self.image_height), dtype=np.uint8)
                    for idx, region_index in enumerate(self.region_list):
                        mask = cv2.imread(
                            mask_dir + '/{}-{}.png'.format(pure_name, region_index), cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(
                            mask, (self.image_height, self.image_width), interpolation=cv2.INTER_NEAREST)
                        index_mask[mask > 100] = idx + 1
                    for i in range(len(self.region_list) + 1):
                        multi_channel_mask[index_mask == i, i] = 255
                else:
                    multi_channel_mask = np.zeros(
                        (self.image_height, self.image_width, len(
                            self.region_list)), dtype=np.uint8)
                    for idx, region_index in enumerate(self.region_list):
                        mask = cv2.imread(
                            mask_dir + '/{}-{}.png'.format(pure_name, region_index), cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(
                            mask, (self.image_height, self.image_width), interpolation=cv2.INTER_NEAREST)
                        multi_channel_mask[mask > 100, idx] = 255
                mask_list.append(multi_channel_mask)

        return np.asarray(image_list), np.asarray(mask_list), file_name_list

    def __len__(self):
        return int(self.image.shape[0] / self.batch_size)

    def __getitem__(self, idx):

        if self.no_ref_mask:
            batch_x = np.zeros(
                shape=(
                    self.batch_size,
                    self.image_width,
                    self.image_height,
                    1),
                dtype=np.float32)
        else:
            batch_x = np.zeros(
                shape=(
                    self.batch_size,
                    self.image_width,
                    self.image_height,
                    self.channel_number),
                dtype=np.float32)

        if not self.no_mask:
            if self.use_background:
                batch_y = np.zeros(
                    shape=(
                        self.batch_size,
                        self.image_width,
                        self.image_height,
                        self.channel_number),
                    dtype=np.uint8)
            else:
                batch_y = np.zeros(
                    shape=(
                        self.batch_size,
                        self.image_width,
                        self.image_height,
                        self.channel_number - 1),
                    dtype=np.uint8)

        for i in range(self.batch_size):
            batch_x[i, :, :, 0] = self.image[idx * self.batch_size + i]
            if not self.no_ref_mask:
                batch_x[i, :, :, 1:] = self.ref_mask
            if not self.no_mask:
                batch_y[i, :, :, :] = self.mask[idx * self.batch_size + i]

        #show_mask = utils.drawMultiRegionMultiChannel(self.mask[idx*self.batch_size + i])
        #cv2.imwrite('../data/augmentation/test_{}.png'.format(idx), self.image[idx*self.batch_size + i])
        #cv2.imwrite('../data/augmentation/test_{}_mask.png'.format(idx), show_mask)

        batch_x = batch_x / 255.0
        #batch_y = batch_y/255.0
        if not self.no_mask:
            batch_y[batch_y < 100] = 0
            batch_y[batch_y > 100] = 1
            return batch_x, batch_y
        return batch_x
