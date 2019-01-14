# code from: https://github.com/pambros/CNN-2D-X-Ray-Catheter-Detection

import keras

from keras.models import Model, Sequential, model_from_json, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, Deconvolution2D, MaxPooling2D, UpSampling2D, Cropping1D, Conv3D, \
    MaxPooling3D, UpSampling3D
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np

print(keras.__version__)


class UNET(object):

    def ResidualBlock(self, _inputs, _filter_num, _n_conv=2, _kernel_size=3, _stride=1, _activation='relu',
                      _padding='same', _BN=True, _data_format='channels_last'):
        cut = _inputs
        for i in range(_n_conv):
            tem = Conv2D(_filter_num, _kernel_size, padding=_padding)(_inputs)
            if _BN:
                if _data_format == 'channels_last':
                    tem = BatchNormalization(axis=-1)(tem)
                else:
                    tem = BatchNormalization(axis=1)(tem)
            if i != _n_conv - 1:
                tem = Activation(_activation)(tem)
            if i == _n_conv - 1:
                tem = keras.layers.Add()([tem, cut])
        return tem

    def DownsampleBlock(self, _inputs, _nbFilters, _kernel_size=2, _stride=2, _mode='conv', _activation='relu',
                        _padding='same', _BN=True, _data_format='channels_last'):
        if _mode == 'max_pool':
            x = MaxPooling2D(
                pool_size=(
                    _kernel_size,
                    _kernel_size),
                data_format=_data_format)(_inputs)
            # TODO max pool does not work with residual, make it work

        else:
            x = Conv2D(
                _nbFilters,
                kernel_size=_kernel_size,
                padding=_padding,
                strides=_stride)(_inputs)
            if _BN:
                if _data_format == 'channels_last':
                    x = BatchNormalization(axis=-1)(x)
                else:
                    x = BatchNormalization(axis=1)(x)
            x = Activation(_activation)(x)
        return x

    def UpsampleBlock(self, _inputs, _nbFilters, _kernel_size=2, _stride=1, _activation='relu',
                      _padding='same', _BN=True, _data_format='channels_last'):

        x = UpSampling2D(size=(2, 2), data_format=_data_format)(_inputs)
        x = Conv2D(
            _nbFilters,
            kernel_size=_kernel_size,
            strides=_stride,
            padding=_padding)(x)
        if _BN:
            if _data_format == 'channels_last':
                x = BatchNormalization(axis=-1)(x)
            else:
                x = BatchNormalization(axis=1)(x)
        x = Activation(_activation)(x)
        return x

    def BuildUnet(self, _input_shape, _filter_num, _depth, dropout=True, _kernel_size=3,
                  _stride=2, _nchannel=1, _data_format='channels_last', use_softmax=False):

        cut = []  # list used to save all cuts
        filters = []  # list used to save all filter numbers
        filter_num = _filter_num

        if _data_format == 'channels_last':
            axis = -1
        else:
            axis = 1

        inputs = Input(shape=_input_shape)
        x = GaussianNoise(0.03)(inputs)
        x = Conv2D(_filter_num, 1, padding='same', data_format=_data_format)(x)

        #  Down sampling part
        for i in range(_depth):
            filters.append(filter_num)
            x = self.ResidualBlock(_inputs=x, _filter_num=filter_num)
            cut.append(x)
            filter_num *= 2
            x = self.DownsampleBlock(x, filter_num)
            if i >= _depth - 2 and dropout:
                x = Dropout(0.5)(x)

        print(filters)
        # One left over bottom block
        x = self.ResidualBlock(_inputs=x, _filter_num=filter_num)

        #  Up sampling part
        for i in range(_depth):
            filter_num = filter_num // 2
            x = self.UpsampleBlock(x, filter_num)
            shortcut = cut[-(i + 1)]
            x = keras.layers.concatenate([x, shortcut], axis=axis)
            x = self.ResidualBlock(
                _inputs=x, _filter_num=2 * filters[-(i + 1)])

        # Output
        if use_softmax:
            # for cross entropy
            outputs = Conv2D(_nchannel, (1, 1), activation='softmax')(x)
        else:
            # for iou loss
            outputs = Conv2D(_nchannel, (1, 1), activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model
