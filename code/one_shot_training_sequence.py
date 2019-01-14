import numpy as np
import cv2
# from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import Sequence
from imgaug import augmenters as iaa
import random
import utils


class OneShotTrainingSequenceMultiRegion(Sequence):
    def __init__(self, image, mask, ref_mask, batch_size=4, use_background=True, no_ref_mask=False, augmentation_number=200,
                 image_width=512, image_height=512):

        self.image = image
        self.mask = mask
        self.ref_mask = ref_mask
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.channel_number = mask.shape[-1]
        self.use_background = use_background
        self.no_ref_mask = no_ref_mask
        self.seq = iaa.Sequential([
            iaa.Add((-50, 50))
        ])
        self.augmentation_number = augmentation_number

    def __len__(self):
        return int(self.augmentation_number / float(self.batch_size))

    def __getitem__(self, idx):
        if self.no_ref_mask:
            # one channel
            batch_x = np.zeros(
                shape=(
                    self.batch_size,
                    self.image_width,
                    self.image_height,
                    1),
                dtype=np.float32)
        else:
            if self.use_background:
                batch_x = np.zeros(
                    shape=(
                        self.batch_size,
                        self.image_width,
                        self.image_height,
                        self.channel_number),
                    dtype=np.float32)
            else:
                batch_x = np.zeros(
                    shape=(
                        self.batch_size,
                        self.image_width,
                        self.image_height,
                        self.channel_number + 1),
                    dtype=np.float32)
        # label
        batch_y = np.zeros(
            shape=(
                self.batch_size,
                self.image_width,
                self.image_height,
                self.channel_number),
            dtype=np.uint8)

        for i in range(self.batch_size):
            # augmentation
            if random.random() < -0.7:
                angle = 0
            else:
                #angle = random.uniform(-25, 25)
                angle = random.uniform(-10, 10)
                #angle = random.uniform(-45, 45)
            if random.random() < -0.5:
                scale = 1.0
            else:
                #scale = random.uniform(0.9, 1.1)
                scale = random.uniform(0.95, 1.05)
            tmp_rotation_matrix = cv2.getRotationMatrix2D((self.image_width / 2, self.image_height / 2),
                                                          angle=angle, scale=scale)
            rotation_matrix = np.eye(3, dtype=np.float32)
            rotation_matrix[0:2, :] = tmp_rotation_matrix

            shearing_matrix = np.eye(3, dtype=np.float32)
            if random.random() < -0.5:
                shearing_matrix[0, 1] = 0.0
                shearing_matrix[1, 0] = 0.0
            else:
                shearing_matrix[0, 1] = random.uniform(-0.005, 0.005)
                shearing_matrix[1, 0] = random.uniform(-0.005, 0.005)

            translation_matrix = np.eye(3, dtype=np.float32)
            translation_matrix[0, 2] = random.randint(-25, 25)
            translation_matrix[1, 2] = random.randint(-25, 25)
            #translation_matrix[0,2] = random.randint(-10, 10)
            #translation_matrix[1,2] = random.randint(-10, 10)

            transform_matrix = np.matmul(
                translation_matrix, np.matmul(
                    shearing_matrix, rotation_matrix))

            transformed_image = cv2.warpPerspective(self.image, transform_matrix, (self.image_width, self.image_height),
                                                    flags=cv2.INTER_LINEAR, borderValue=(255))
            transformed_mask = np.zeros(
                (self.image_height, self.image_width, self.mask.shape[-1]), dtype=np.uint8)

            if self.use_background:
                # need a index mask to deal with the background
                transformed_index_mask = np.zeros(
                    (self.image_height, self.image_width), dtype=np.uint8)
                for j in range(1, self.mask.shape[-1]):
                    temp_mask = cv2.warpPerspective(self.mask[:, :, j], transform_matrix, (self.image_width, self.image_height),
                                                    flags=cv2.INTER_NEAREST, borderValue=(0))
                    transformed_index_mask[temp_mask > 100] = j

                for j in range(self.mask.shape[-1]):
                    transformed_mask[transformed_index_mask == j, j] = 255
            else:
                # no background, deal with this directly
                for j in range(self.mask.shape[-1]):
                    temp_mask = cv2.warpPerspective(self.mask[:, :, j], transform_matrix, (self.image_width, self.image_height),
                                                    flags=cv2.INTER_NEAREST, borderValue=(0))
                    transformed_mask[temp_mask > 100, j] = 255

            aug_image = self.seq.augment_image(transformed_image)
            #show_mask = utils.drawMultiRegionMultiChannel(transformed_mask)
            ##aug_image = cv2.equalizeHist(aug_image)
            #cv2.imwrite('../data/augmentation/{}_img.png'.format(i), aug_image)
            #cv2.imwrite('../data/augmentation/{}_mask.png'.format(i), show_mask)

            batch_x[i, :, :, 0] = aug_image
            if not self.no_ref_mask:
                batch_x[i, :, :, 1:] = self.ref_mask
            batch_y[i, :, :, :] = transformed_mask

        batch_x = batch_x / 255.0
        batch_y[batch_y < 100] = 0
        batch_y[batch_y >= 100] = 1

        return batch_x, batch_y
