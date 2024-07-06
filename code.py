from google.colab import drive
drive.mount('/content/drive')
orgpath = '/content/drive/MyDrive/[file_path]/'

import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from sklearn.metrics import confusion_matrix, jaccard_score
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
import tensorflow.keras.backend as K
import tensorflow as tf

FOLDER_PATH = orgpath + 'Processed_Dataset/'#processed_dataset is the name of the folder
image_size = 256
batch_size = 64
epochs = 50


train_path = FOLDER_PATH + "train/"
validation_path = FOLDER_PATH + "val/"
test_path = FOLDER_PATH + "test"

class DataGen(Sequence):
    def __init__(self, ids, path, batch_size=2, image_size=256):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __load__(self, id_name):
        image_path = os.path.join(self.path, "images", "img", id_name)
        mask_path = os.path.join(self.path, "mask", "img", id_name)
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            return None, None
        image = cv2.resize(image, (self.image_size, self.image_size)) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.size == 0:
            return None, None
        mask = cv2.resize(mask, (self.image_size, self.image_size)) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        return image, mask

    def __getitem__(self, index):
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        images, masks = [], []
        for id_name in files_batch:
            img, msk = self.__load__(id_name)
            if img is not None and msk is not None:
                images.append(img)
                masks.append(msk)
        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        np.random.shuffle(self.ids)

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

  def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def F1(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def dice_coef(y_true, y_pred, smooth = 0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def load_test_data(batch_size=10):
    test_ids = os.listdir(os.path.join(test_path, "images", "img"))
    gen = DataGen(test_ids, test_path, batch_size=batch_size, image_size=image_size)
    print("Test Data Loaded")
    return gen, len(test_ids)

def scheduler(epoch, lr):
    if epoch < 30:
        lr = 0.001
        return lr
    if epoch < 50:
        return 0.0005
    return 0.0001

def visualize_output(x, y, y_pred, tot, idx, fig, title_flag=False):
    contours1, hierarchy = cv2.findContours(y[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2, hierarchy = cv2.findContours(y_pred[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(x[0], contours1, -1, (0, 255, 0), 3)
    cv2.drawContours(x[0], contours2, -1, (255, 0, 0), 3)
    ax = fig.add_subplot(tot, 5, idx)
    ax.imshow(np.reshape(x[0], (image_size, image_size, 3)))

def evaluate_model(gen, model):
    results = {}
    y_true_all = []
    y_pred_all = []
    num_tests = len(gen)
    for i in range(num_tests):
        x, y_true = gen[i]
        y_pred_prob = model.predict(x)
        y_pred = (y_pred_prob > 0.8).astype(int)
        y_true = (y_true > 0.5).astype(int)
        y_true_all.extend(y_true.flatten())
        y_pred_all.extend(y_pred.flatten())
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    iou = jaccard_score(y_true_all, y_pred_all)
    out = model.evaluate(gen, steps=num_tests)
    last_metric = out[-1]
    results = {
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'IoU': iou,
        'Eval Metric': last_metric
    }
    print(results)

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Reshape, Flatten, Dense

def cut_in_detection_model(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Backbone (simplified for demonstration)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Flatten the output from convolutional layers
    x = Flatten()(x)
    
    # Dense layers for classification
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(num_classes, activation='sigmoid')(x)

    # Define model
    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

# Example usage:
input_shape = (416, 416, 3)  # example input shape
num_classes = 1  # binary classification (cut-in or not)
model = cut_in_detection_model(input_shape, num_classes)
model.summary()

train_ids = os.listdir(os.path.join(train_path, "images"))
val_ids = os.listdir(os.path.join(validation_path, "images"))
test_ids = os.listdir(os.path.join(test_path, "images"))

train_gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
val_gen = DataGen(val_ids, validation_path, batch_size=batch_size, image_size=image_size)
test_gen = DataGen(test_ids, test_path, batch_size=batch_size, image_size=image_size)

total_train_images = len(train_gen) * batch_size
total_val_images = len(val_gen) * batch_size
total_test_images = len(test_gen) * batch_size

print(f"Total training ids: {len(train_ids)}")
print(f"Total validation ids: {len(val_ids)}")
print(f"Total test ids: {len(test_ids)}")

print(f"Total training images: {total_train_images}")
print(f"Total validation images: {total_val_images}")
print(f"Total test images: {total_test_images}")

# Load training and validation data
train_ids = os.listdir(os.path.join(train_path, "images"))
val_ids = os.listdir(os.path.join(validation_path, "images"))

train_gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
val_gen = DataGen(val_ids, validation_path, batch_size=batch_size, image_size=image_size)

# Callbacks
checkpoint = ModelCheckpoint('unet_mhsa_best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
lr_scheduler = LearningRateScheduler(scheduler)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[checkpoint, lr_scheduler, reduce_lr]
)

model.save(orgpath + 'vehicle cut in')
print("Model saved successfully!")

model_path = orgpath + 'vci_final_model'
model = tf.keras.models.load_model(model_path, custom_objects={'bce_dice_loss': bce_dice_loss, 'iou': iou, 'F1': F1, 'recall': recall, 'precision': precision, 'dice_coef': dice_coef})

# Load test data
test_gen, test_len = load_test_data(batch_size=batch_size)

# Evaluate the model
evaluate_model(test_gen, model)

from google.colab import runtime
runtime.unassign()
