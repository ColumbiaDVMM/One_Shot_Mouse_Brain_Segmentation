from U_net import *
from keras import backend as K
import os
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import keras.losses

import tensorflow as tf

'''
All metrics are for binary mode only, any other situation will cause unpredictable result
'''


#%% All  metrics(dice, iou, precision, recall)
def Dice_Coef(y_true, y_pred):

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    dice = (2 * intersection) / (sum_ + K.epsilon())

    return dice

def Dice_Coef_Multi_Region(y_true, y_pred):

    channel_number = K.int_shape(y_pred)[-1]
    y_true = K.reshape(y_true, (-1, channel_number))
    y_pred = K.reshape(y_pred, (-1, channel_number))

    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis = 0), axis = 0)
    sum_ = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis = 0), axis = 0)

    dice = (2 * intersection) / (sum_ + K.epsilon())
    dice = K.mean(dice)
    return dice

def Iou_Coef_Multi_Region(y_true, y_pred):
    #y_pred = K.round(K.clip(y_pred, 0, 1))
    #y_pred = y_pred>0.5
    channel_number = K.int_shape(y_pred)[-1]
    y_true = K.reshape(y_true, (-1, channel_number))
    y_pred = K.reshape(y_pred, (-1, channel_number))

    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis = 0), axis = 0)
    sum_ = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis = 0), axis = 0)
    jac = (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    jac = K.mean(jac)
    return jac

def Iou_Coef(y_true, y_pred):
    #y_pred = K.round(K.clip(y_pred, 0, 1))
    #y_pred = y_pred>0.5
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    jac = (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    return jac

def Iou_Metric_Multi_Region_with_Background(y_true, y_pred):
    channel_number = K.int_shape(y_pred)[-1]

    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    # with background, remove the background channel
    jac = 0 
    for i in range(1, channel_number):
        tmp_y_true = K.cast(K.equal(y_true,i),'float32')
        tmp_y_pred = K.cast(K.equal(y_pred,i),'float32')
        intersection = K.sum(K.abs(tmp_y_true * tmp_y_pred))
        sum_ = K.sum(K.abs(tmp_y_true) + K.abs(tmp_y_pred))
        jac += (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    jac = jac/(channel_number-1)
    return jac

def Iou_Metric_Multi_Region(y_true, y_pred):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    y_pred = K.cast(y_pred>0.5, 'float32')

    channel_number = K.int_shape(y_pred)[-1]
    y_true = K.reshape(y_true, (-1, channel_number))
    y_pred = K.reshape(y_pred, (-1, channel_number))

    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis = 0), axis = 0)
    sum_ = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis = 0), axis = 0)
    jac = (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    jac = K.mean(jac)
    return jac

def Iou_Metric(y_true, y_pred):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    jac = (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    return jac

def get_iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    sum_ = np.sum(y_true)+np.sum(y_pred)
    jac = (intersection + 1e-7) / (sum_ - intersection + 1e-7)
    return jac

def Precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def Recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def Hybrid_metric(y_true, y_pred, weight=(1, 1, 1)):
    # hybrid metrics with iou, precision, recall
    iou = Iou_Coef(y_true, y_pred)
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)

    w1, w2, w3 = weight
    # equal weight for three metrics
    print('iou:', iou, 'precision', precision, 'recall', recall)
    return (iou*w1 + precision*w2 + recall*w3) / (w1 + w2 + w3)


#%% loss
def cross_entropy(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred)
    loss = -K.sum(loss, -1)
    return loss


def weighted_categorical_crossentropy_loss(y_true, y_pred):
    channel_number = K.int_shape(y_pred)[-1]
    # equal weight
    # weights = [0.25,0.25,0.25,0.25,0.25]
    weights = [1]*channel_number

    # weight proportion to reciprocal of area
    # weights = [0.823, 0.031, 0.040, 0.037, 0.068]

    # weight proportion to area
    # weights = [0.012, 0.319, 0.249, 0.272, 0.147]

    weights = K.variable(weights)
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)

    return loss 


def Dice_Coef_Loss(y_true, y_pred):
    return -Dice_Coef(y_true, y_pred)

def Dice_Coef_Multi_Region_Loss(y_true, y_pred):
    return -Dice_Coef_Multi_Region(y_true, y_pred)


def Iou_Coef_Loss(y_true, y_pred):
    return -Iou_Coef(y_true, y_pred)


def Iou_Coef_Multi_Region_Loss(y_true, y_pred):
    return -Iou_Coef_Multi_Region(y_true, y_pred)


def Precision_loss(y_true, y_pred):
    return -Precision(y_true, y_pred)


def Recall_loss(y_true, y_pred):
    return -Recall(y_true, y_pred)


def Hybrid_loss(y_true, y_pred):
    return -Hybrid_metric(y_true, y_pred)


#%%
def Compile(model, optimizer='SGD', lr=0.01, metric='iou', loss='dice_loss', pre_weights=False):

    if pre_weights:
        model.load_weights(pre_weights)

    if optimizer == 'SGD':
        optimizer = SGD(lr=lr, decay=5e-4, momentum=0.99)
    if optimizer == 'Adam':
        optimizer = Adam(lr=lr, decay=5e-4)
    
    if loss == 'dice':
        loss = Dice_Coef_Loss
    elif loss == 'dice-multi-region':
        loss = Dice_Coef_Multi_Region_Loss
    elif loss == 'iou':
        loss = Iou_Coef_Loss
    elif loss == 'iou-multi-region':
        loss = Iou_Coef_Multi_Region_Loss
    elif loss == 'weighted_categorical_cross_entropy':
        loss = weighted_categorical_crossentropy_loss
    elif loss == 'cross_entropy':
        loss = keras.losses.categorical_crossentropy
    elif loss == 'precision':
        loss = Precision_loss
    elif loss == 'recall':
        loss = Recall_loss
    elif loss == 'hybrid':
        loss = Hybrid_loss

    if metric == 'dice':
        metric = Dice_Coef
        metric_2 = Iou_Metric_Multi_Region
    elif metric == 'iou':
        metric = Iou_Coef
        metric_2 = Iou_Metric
    elif metric == 'iou-multi-region-with-background':
        metric = Iou_Coef_Multi_Region
        metric_2 = Iou_Metric_Multi_Region_with_Background
    elif metric == 'iou-multi-region':
        metric = Iou_Coef_Multi_Region
        metric_2 = Iou_Metric_Multi_Region

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric, metric_2])


#%%
def Train(model, images, masks, weight_path, tensorboard_path, tensorboard_name, \
        monitor='loss', steps_per_epoch=4, epochs=50, name="best"):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    # modelCheckpoint, save best only
    modelCheckpoint = ModelCheckpoint(weight_path + "/" + name + ".h5", monitor=monitor, save_best_only=True,
                                      save_weights_only=True)

    tensorboard = TensorBoard(log_dir=tensorboard_path + '/' + tensorboard_name, batch_size=4)

    model.fit(
        images,
        masks,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=[modelCheckpoint, tensorboard],
    )
    return model


#%%
def TrainGenerator(model, train_generator, weight_path, tensorboard_path,\
        tensorboard_name, monitor='loss', steps_per_epoch=4, epochs=50, \
        name="bestTrainWeight", validation=None):

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    # modelCheckpoint, save best only
    modelCheckpoint = ModelCheckpoint(weight_path + "/" + name + ".h5", monitor=monitor, save_best_only=True,
                                      save_weights_only=True)

    tensorboard = TensorBoard(log_dir=tensorboard_path + '/' + tensorboard_name, batch_size=4)

    model.fit_generator(
        train_generator,
        validation_data=validation,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=[modelCheckpoint, tensorboard],
    )

    return model


#%%
def Save_Model(model, path, model_name):
    if not os.path.exists(path):
        os.makedirs(path)

    model.save(path + '/' + model_name + '.h5')

